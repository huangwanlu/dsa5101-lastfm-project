"""Fixed staged, validation-only model selection for ``lastfm-v1``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
from time import perf_counter
from typing import Mapping, Sequence
import warnings

import implicit
import numpy as np
import pandas as pd

from .als import ALS_GRID, ALSConfig, ALSModel, fit_implicit_als
from .evaluation_ledger import EvaluationPhaseLedger
from .popularity import BETA_VALUES, CommunityPopularity, fit_community_popularity, validate_complete_sparse_labels
from .recommendation import (
    MetricEvaluation,
    als_candidate_scores,
    community_candidate_scores,
    evaluate_ranker,
    hybrid_candidate_scores,
    ndcg_at_k,
    rank_scores,
    recall_at_k,
)


HYBRID_LAMBDAS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class ValidationRecord:
    stage: str
    parameters: Mapping[str, float | int]
    ndcg_at_10: float
    recall_at_10: float
    eligible_users: int
    failed_users: tuple[int, ...]
    runtime_seconds: float
    warnings: tuple[str, ...] = ()

    @property
    def successful(self) -> bool:
        return not self.failed_users and np.isfinite(self.ndcg_at_10)


@dataclass(frozen=True)
class ValidationSelection:
    records: tuple[ValidationRecord, ...]
    best_als: ValidationRecord
    best_community: ValidationRecord
    best_hybrid: ValidationRecord
    total_runtime_seconds: float
    metadata: Mapping[str, object]
    h3_estimable: bool


def _selection_key(record: ValidationRecord) -> tuple[object, ...]:
    parameters = record.parameters
    return (
        -record.ndcg_at_10,
        int(parameters.get("factors", -1)),
        -float(parameters.get("regularization", -np.inf)),
        float(parameters.get("alpha", -np.inf)),
        float(parameters.get("beta", -np.inf)),
        float(parameters.get("lambda", -np.inf)),
        tuple(sorted(parameters.items())),
    )


def select_best(records: Sequence[ValidationRecord], *, stage: str) -> ValidationRecord:
    candidates = [record for record in records if record.stage == stage and record.successful]
    if not candidates:
        raise RuntimeError(f"no successful {stage} validation configurations")
    return min(candidates, key=_selection_key)


def _record(
    stage: str,
    parameters: Mapping[str, float | int],
    evaluation: MetricEvaluation | None,
    runtime: float,
    captured_warnings: Sequence[str],
    failure: str | None = None,
) -> ValidationRecord:
    warnings_out = list(captured_warnings)
    if failure is not None:
        warnings_out.append(failure)
    if evaluation is None:
        return ValidationRecord(stage, dict(parameters), np.nan, np.nan, 0, (), runtime, tuple(warnings_out))
    return ValidationRecord(
        stage,
        dict(parameters),
        evaluation.ndcg_at_10,
        evaluation.recall_at_10,
        len(evaluation.eligible_user_ids),
        evaluation.failed_user_ids,
        runtime,
        tuple(warnings_out),
    )


def _evaluate_als(ledger: EvaluationPhaseLedger, model: ALSModel) -> MetricEvaluation:
    return evaluate_ranker(
        ledger,
        lambda user_id, candidates: als_candidate_scores(model, user_id, candidates),
    )


def _evaluate_community(
    ledger: EvaluationPhaseLedger, profile: CommunityPopularity, membership
) -> MetricEvaluation:
    return evaluate_ranker(
        ledger,
        lambda user_id, candidates: community_candidate_scores(
            profile, membership, user_id, candidates
        ),
    )


def _evaluate_hybrid_lambdas(
    ledger: EvaluationPhaseLedger,
    model: ALSModel,
    profile: CommunityPopularity,
    membership,
) -> dict[float, MetricEvaluation]:
    """Evaluate all fixed lambdas while sharing identical raw components."""
    users = tuple(ledger.table.loc[ledger.table["metric_eligible"], "user_id"].astype(int))
    rows: dict[float, list[tuple[int, float, float]]] = {value: [] for value in HYBRID_LAMBDAS}
    failures: dict[float, list[int]] = {value: [] for value in HYBRID_LAMBDAS}
    counts: dict[float, dict[str, int]] = {value: {} for value in HYBRID_LAMBDAS}
    for user_id in users:
        candidates, relevance = ledger.candidate_relevance(user_id)
        try:
            mf = als_candidate_scores(model, user_id, candidates)
            community = community_candidate_scores(profile, membership, user_id, candidates)
        except (FloatingPointError, ValueError, KeyError):
            for value in HYBRID_LAMBDAS:
                failures[value].append(user_id)
            continue
        for value in HYBRID_LAMBDAS:
            try:
                ranking = rank_scores(
                    hybrid_candidate_scores(mf, community, lambda_=value), limit=10
                )
            except (FloatingPointError, ValueError, KeyError):
                failures[value].append(user_id)
                continue
            for state, count in ranking.state_counts.items():
                counts[value][state] = counts[value].get(state, 0) + count
            if ranking.failed:
                failures[value].append(user_id)
                continue
            ranked = ranking.artist_ids
            rows[value].append(
                (user_id, ndcg_at_k(ranked, relevance), recall_at_k(ranked, relevance))
            )
    results: dict[float, MetricEvaluation] = {}
    for value in HYBRID_LAMBDAS:
        per_user = pd.DataFrame(
            rows[value], columns=["user_id", "ndcg_at_10", "recall_at_10"]
        ).astype({"user_id": "int64", "ndcg_at_10": "float64", "recall_at_10": "float64"})
        successful = not failures[value] and bool(rows[value])
        results[value] = MetricEvaluation(
            float(per_user["ndcg_at_10"].mean()) if successful else np.nan,
            float(per_user["recall_at_10"].mean()) if successful else np.nan,
            per_user,
            users,
            tuple(failures[value]),
            counts[value],
        )
    return results


def run_validation_selection(
    ledger: EvaluationPhaseLedger,
    membership,
    *,
    show_progress: bool = False,
) -> ValidationSelection:
    """Run the exact 8/3/9 staged grid using validation fitting data only."""
    if ledger.phase != "validation":
        raise ValueError("model selection may only use a validation-phase ledger")
    fixed_labels = ledger.table.loc[
        ledger.table["raw_evaluation_eligible"], ["user_id", "sparse"]
    ]
    validate_complete_sparse_labels(ledger, fixed_labels)
    started = perf_counter()
    records: list[ValidationRecord] = []
    als_models: dict[ALSConfig, ALSModel] = {}

    for config in ALS_GRID:
        then = perf_counter()
        messages: list[str] = []
        evaluation = None
        failure = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = fit_implicit_als(
                    ledger.fitting_interactions, config, show_progress=show_progress
                )
                evaluation = _evaluate_als(ledger, model)
            messages = [str(item.message) for item in caught]
            als_models[config] = model
        except Exception as exc:  # record a failed configuration; never substitute it
            failure = f"{type(exc).__name__}: {exc}"
        records.append(_record("als", asdict(config), evaluation, perf_counter() - then, messages, failure))

    best_als = select_best(records, stage="als")
    selected_config = ALSConfig(
        factors=int(best_als.parameters["factors"]),
        regularization=float(best_als.parameters["regularization"]),
        alpha=float(best_als.parameters["alpha"]),
    )
    selected_model = als_models[selected_config]

    community_profiles: dict[float, CommunityPopularity] = {}
    for beta in BETA_VALUES:
        then = perf_counter()
        evaluation = None
        failure = None
        try:
            profile = fit_community_popularity(
                ledger.fitting_interactions, membership, beta=beta
            )
            evaluation = _evaluate_community(ledger, profile, membership)
            community_profiles[beta] = profile
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        records.append(_record("community", {"beta": beta}, evaluation, perf_counter() - then, (), failure))

    best_community = select_best(records, stage="community")

    for beta in BETA_VALUES:
        then = perf_counter()
        evaluations: dict[float, MetricEvaluation] = {}
        failure = None
        try:
            evaluations = _evaluate_hybrid_lambdas(
                ledger, selected_model, community_profiles[beta], membership
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        shared_runtime = perf_counter() - then
        for lambda_ in HYBRID_LAMBDAS:
            records.append(
                _record(
                    "hybrid",
                    {"beta": beta, "lambda": lambda_},
                    evaluations.get(lambda_),
                    shared_runtime,
                    ("runtime shared across the three lambda evaluations for this beta",),
                    failure,
                )
            )

    best_hybrid = select_best(records, stage="hybrid")
    sparse_values = fixed_labels["sparse"].astype(bool)
    h3_estimable = bool(sparse_values.any() and (~sparse_values).any())
    metadata = {
        "protocol_id": "lastfm-v1",
        "selection_metric": "macro validation NDCG@10",
        "implicit_version": implicit.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "library_input_orientation": "users x artists CSR",
        "confidence_encoding": "CSR data = 1 + alpha * log1p(weight); implicit alpha = 1",
        "als_num_threads": 1,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "fit_scope": "train + train_only_low_history only",
        "test_scored": False,
    }
    return ValidationSelection(
        tuple(records), best_als, best_community, best_hybrid,
        perf_counter() - started, metadata, h3_estimable
    )


__all__ = [
    "HYBRID_LAMBDAS", "ValidationRecord", "ValidationSelection", "run_validation_selection",
    "select_best",
]
