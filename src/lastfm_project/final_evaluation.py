"""Approved final evaluation and inference for the ``lastfm-v1`` protocol.

The public helpers in this module are deliberately usable on small synthetic
data sets.  :func:`run_final_analysis` is the only entry point that reads the
real archive and it verifies the P4 selection record before inspecting the
test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from .als import ALSConfig, ALSModel, fit_implicit_als
from .communities import detect_communities, shuffle_community_assignments
from .data_loading import load_hetrec
from .evaluation_ledger import EvaluationPhaseLedger, build_phase_ledger
from .popularity import (
    CommunityPopularity,
    GlobalPopularity,
    fit_community_popularity,
    fit_global_popularity,
    validate_complete_sparse_labels,
)
from .recommendation import (
    INVALID_NUMERICAL,
    MISSING_COMPONENT,
    MISSING_REPRESENTATION,
    VALID,
    ZERO_PROBABILITY,
    CandidateScores,
    als_candidate_scores,
    community_candidate_scores,
    hybrid_candidate_scores,
    ndcg_at_k,
    rank_scores,
    recall_at_k,
)
from .splitting import PROTOCOL_ID, SPLIT_SEED, compute_sparse_labels, split_interactions


SELECTED_ALS = ALSConfig(32, 0.10, 1.0, iterations=20, seed=5101)
SELECTED_COMMUNITY_BETA = 1000.0
SELECTED_HYBRID_BETA = 100.0
SELECTED_HYBRID_LAMBDA = 0.25
BOOTSTRAP_RESAMPLES = 2000
H1_STRATA = ((5, 10, "5-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, None, "51+"))


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    resamples: int
    seed: int
    user_count: int


@dataclass(frozen=True)
class MethodEvaluation:
    method: str
    candidate_policy: str
    ndcg_at_10: float
    recall_at_10: float
    coverage_at_10: float
    per_user: pd.DataFrame
    eligible_user_ids: tuple[int, ...]
    failed_user_ids: tuple[int, ...]
    state_counts: Mapping[str, int]
    rankings_with_missing_scores: int
    distinct_recommended_artists: int
    universe_size: int
    runtime_seconds: float


@dataclass(frozen=True)
class H1Result:
    anchors: pd.DataFrame
    candidate_anchor_counts: Mapping[str, int]
    peer_reuse: pd.DataFrame
    mean_contrast: float
    jackknife_se: float
    ci_lower: float
    ci_upper: float
    jackknife_users: int
    jackknife_omitted: int


@dataclass(frozen=True)
class FinalAnalysis:
    comparison: pd.DataFrame
    robustness: pd.DataFrame
    h2: BootstrapResult
    h2_interpretation: str
    h1: H1Result
    h3: Mapping[str, object]
    provenance: Mapping[str, object]
    warnings: tuple[str, ...]
    runtime_seconds: float
    output_directory: Path


def verify_selected_parameters(record_path: str | Path) -> None:
    """Fail unless the human-readable P4 record contains the selected choices."""
    text = Path(record_path).read_text(encoding="utf-8")
    checks = {
        "ALS": r"ALS:\s*factors\s*32,\s*regularization\s*0\.10,\s*alpha\s*1,\s*iterations\s*20,\s*seed\s*5101",
        "Community": r"Community:\s*beta\s*1,?000",
        "Hybrid": r"Hybrid:.*beta\s*100\s+and\s+lambda\s*0\.25",
    }
    missing = [name for name, pattern in checks.items() if re.search(pattern, text, re.IGNORECASE) is None]
    if missing:
        raise ValueError(f"P4 record does not match selected parameters for: {', '.join(missing)}")


def paired_ndcg_bootstrap(
    als_per_user: pd.DataFrame,
    hybrid_per_user: pd.DataFrame,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SPLIT_SEED,
) -> BootstrapResult:
    """Calculate the fixed paired Hybrid-minus-ALS user bootstrap."""
    required = {"user_id", "ndcg_at_10"}
    if not required.issubset(als_per_user.columns) or not required.issubset(hybrid_per_user.columns):
        raise ValueError("paired inputs must contain user_id and ndcg_at_10")
    if als_per_user["user_id"].duplicated().any() or hybrid_per_user["user_id"].duplicated().any():
        raise ValueError("paired inputs contain duplicate users")
    als_ids = tuple(als_per_user["user_id"].astype(int))
    hybrid_ids = tuple(hybrid_per_user["user_id"].astype(int))
    if als_ids != hybrid_ids:
        raise ValueError("paired-user mismatch between ALS and hybrid")
    if not als_ids:
        raise ValueError("paired bootstrap requires at least one user")
    differences = (
        hybrid_per_user["ndcg_at_10"].to_numpy(dtype=np.float64)
        - als_per_user["ndcg_at_10"].to_numpy(dtype=np.float64)
    )
    if not np.isfinite(differences).all():
        raise FloatingPointError("paired NDCG differences must be finite")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(resamples, len(differences)))
    bootstrap = differences[draws].mean(axis=1)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return BootstrapResult(
        float(differences.mean()), float(lower), float(upper), resamples, seed, len(differences)
    )


def _stratum(count: int) -> str:
    for lower, upper, label in H1_STRATA:
        if count >= lower and (upper is None or count <= upper):
            return label
    raise ValueError(f"eligible training-history count is outside H1 strata: {count}")


def _hashed_peer(anchor: int, peers: Iterable[int], kind: str, *, seed: int) -> int:
    choices = sorted(
        (int(peer) for peer in peers),
        key=lambda peer: (
            sha256(f"{seed}|H1|{anchor}|{kind}|{peer}".encode("ascii")).digest(),
            peer,
        ),
    )
    if not choices:
        raise ValueError("cannot select from an empty H1 peer set")
    return choices[0]


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def descriptive_h1_jackknife(anchors: pd.DataFrame) -> tuple[float, float, float, int, int]:
    """Return SE, normal CI, number of deletion users, and omitted deletions."""
    required = {"anchor_user_id", "within_peer_id", "cross_peer_id", "contrast"}
    if not required.issubset(anchors.columns) or anchors.empty:
        raise ValueError("H1 jackknife requires nonempty complete anchor rows")
    values = anchors["contrast"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("H1 contrasts must be finite")
    users = sorted(
        set(anchors["anchor_user_id"].astype(int))
        | set(anchors["within_peer_id"].astype(int))
        | set(anchors["cross_peer_id"].astype(int))
    )
    estimates: list[float] = []
    omitted = 0
    for user_id in users:
        keep = ~(
            (anchors["anchor_user_id"] == user_id)
            | (anchors["within_peer_id"] == user_id)
            | (anchors["cross_peer_id"] == user_id)
        )
        remaining = anchors.loc[keep, "contrast"]
        if len(remaining) < 2:
            omitted += 1
        else:
            estimates.append(float(remaining.mean()))
    if len(estimates) < 2:
        raise ValueError("fewer than two usable H1 jackknife deletions")
    estimates_array = np.asarray(estimates, dtype=np.float64)
    center = float(estimates_array.mean())
    n = len(estimates_array)
    standard_error = float(np.sqrt((n - 1) / n * np.square(estimates_array - center).sum()))
    estimate = float(values.mean())
    return standard_error, estimate - 1.96 * standard_error, estimate + 1.96 * standard_error, len(users), omitted


def run_h1_analysis(
    split_table: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
) -> H1Result:
    """Run the fixed training-only activity-matched H1 comparison."""
    if membership["user_id"].duplicated().any():
        raise ValueError("membership contains duplicate users")
    eligible = sorted(split_table.loc[split_table["split"] == "train", "user_id"].astype(int).unique())
    labels = membership.set_index("user_id")["community_id"]
    missing = sorted(set(eligible) - set(labels.index.astype(int)))
    if missing:
        raise ValueError(f"missing H1 memberships: {missing[:10]}")
    artist_sets = {
        int(user): frozenset(group["artist_id"].astype(int))
        for user, group in split_table.loc[split_table["split"] == "train"].groupby("user_id", sort=True)
    }
    strata = {user: _stratum(len(artist_sets[user])) for user in eligible}
    candidate_counts = {label: 0 for _, _, label in H1_STRATA}
    rows: list[tuple[object, ...]] = []
    for anchor in eligible:
        within = [
            peer for peer in eligible
            if peer != anchor and strata[peer] == strata[anchor] and int(labels.loc[peer]) == int(labels.loc[anchor])
        ]
        cross = [
            peer for peer in eligible
            if strata[peer] == strata[anchor] and int(labels.loc[peer]) != int(labels.loc[anchor])
        ]
        if not within or not cross:
            continue
        candidate_counts[strata[anchor]] += 1
        within_peer = _hashed_peer(anchor, within, "within", seed=seed)
        cross_peer = _hashed_peer(anchor, cross, "cross", seed=seed)
        within_similarity = _jaccard(artist_sets[anchor], artist_sets[within_peer])
        cross_similarity = _jaccard(artist_sets[anchor], artist_sets[cross_peer])
        rows.append((
            anchor, strata[anchor], within_peer, cross_peer,
            within_similarity, cross_similarity, within_similarity - cross_similarity,
        ))
    anchors = pd.DataFrame(rows, columns=[
        "anchor_user_id", "stratum", "within_peer_id", "cross_peer_id",
        "within_jaccard", "cross_jaccard", "contrast",
    ]).astype({
        "anchor_user_id": "int64", "stratum": "string", "within_peer_id": "int64",
        "cross_peer_id": "int64", "within_jaccard": "float64",
        "cross_jaccard": "float64", "contrast": "float64",
    })
    if anchors.empty:
        raise ValueError("no H1 anchors have both activity-matched peer types")
    peer_ids = pd.concat([anchors["within_peer_id"], anchors["cross_peer_id"]], ignore_index=True)
    peer_reuse = (
        peer_ids.value_counts(sort=False).rename_axis("user_id").rename("use_count").reset_index()
        .sort_values(["use_count", "user_id"], ascending=[False, True], kind="mergesort", ignore_index=True)
        .astype("int64")
    )
    se, lower, upper, users, omitted = descriptive_h1_jackknife(anchors)
    return H1Result(
        anchors, candidate_counts, peer_reuse, float(anchors["contrast"].mean()),
        se, lower, upper, users, omitted,
    )


def full_catalog_ledger(
    primary: EvaluationPhaseLedger, catalog_artist_ids: Sequence[int]
) -> EvaluationPhaseLedger:
    """Expand a test ledger to the audited catalog without changing fit data."""
    if primary.phase != "test":
        raise ValueError("full-catalog sensitivity is final-test only")
    universe = tuple(sorted({int(value) for value in catalog_artist_ids}))
    if len(universe) != len(catalog_artist_ids):
        raise ValueError("catalog artist IDs must be unique")
    if not set(primary.primary_artist_universe) <= set(universe):
        raise ValueError("full catalog does not contain every fitting artist")
    table = primary.table.copy(deep=True)
    universe_set = set(universe)
    for index, row in table.iterrows():
        user_id = int(row["user_id"])
        fit = primary._fit_artists_by_user.get(user_id, frozenset())
        heldout = primary._heldout_artists_by_user.get(user_id, frozenset())
        retrievable = heldout & (universe_set - set(fit))
        table.loc[index, "metric_eligible"] = bool(retrievable)
        table.loc[index, "retrievable_heldout_count"] = len(retrievable)
        table.loc[index, "candidate_count"] = len(universe_set - set(fit))
        # Coldness remains relative to learned fitting representations.
        table.loc[index, "cold_heldout_count"] = len(heldout - set(primary.primary_artist_universe))
    return EvaluationPhaseLedger(
        "test", table, primary.fitting_interactions, universe,
        primary._fit_artists_by_user, primary._heldout_artists_by_user,
    )


def global_candidate_scores(
    profile: GlobalPopularity, candidate_artists: tuple[int, ...]
) -> CandidateScores:
    values = np.fromiter(
        (float(profile.scores.get(int(artist), 0.0)) for artist in candidate_artists),
        dtype=np.float64,
        count=len(candidate_artists),
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite global-popularity score")
    return CandidateScores(candidate_artists, values, (VALID,) * len(candidate_artists))


def evaluate_method(
    method: str,
    candidate_policy: str,
    ledger: EvaluationPhaseLedger,
    scorer: Callable[[int, tuple[int, ...]], CandidateScores],
    *,
    k: int = 10,
) -> MethodEvaluation:
    """Evaluate metrics, coverage, failures, and score states on one ledger."""
    started = perf_counter()
    users = tuple(ledger.table.loc[ledger.table["metric_eligible"], "user_id"].astype(int))
    rows: list[tuple[int, float, float]] = []
    failures: list[int] = []
    state_counts: dict[str, int] = {}
    recommended: set[int] = set()
    rankings_with_missing = 0
    for user_id in users:
        candidates, relevance = ledger.candidate_relevance(user_id)
        try:
            component = scorer(user_id, candidates)
            if component.artist_ids != candidates:
                raise ValueError("scorer did not preserve the ledger candidate set and order")
            ranking = rank_scores(component, limit=k)
        except (FloatingPointError, ValueError, KeyError):
            failures.append(user_id)
            continue
        for state, count in ranking.state_counts.items():
            state_counts[state] = state_counts.get(state, 0) + count
        if ranking.failed:
            failures.append(user_id)
            continue
        if any(state != VALID for state in component.states):
            rankings_with_missing += 1
        ranked = ranking.artist_ids
        recommended.update(ranked)
        rows.append((user_id, ndcg_at_k(ranked, relevance, k=k), recall_at_k(ranked, relevance, k=k)))
    per_user = pd.DataFrame(rows, columns=["user_id", "ndcg_at_10", "recall_at_10"]).astype(
        {"user_id": "int64", "ndcg_at_10": "float64", "recall_at_10": "float64"}
    )
    successful = bool(users) and not failures and len(rows) == len(users)
    return MethodEvaluation(
        method, candidate_policy,
        float(per_user["ndcg_at_10"].mean()) if successful else float("nan"),
        float(per_user["recall_at_10"].mean()) if successful else float("nan"),
        float(len(recommended) / len(ledger.primary_artist_universe)) if successful else float("nan"),
        per_user, users, tuple(failures), state_counts, rankings_with_missing,
        len(recommended), len(ledger.primary_artist_universe), perf_counter() - started,
    )


def _result_row(result: MethodEvaluation, ledger: EvaluationPhaseLedger) -> dict[str, object]:
    eligible = ledger.table.loc[ledger.table["raw_evaluation_eligible"]]
    candidate_counts = ledger.table.loc[ledger.table["metric_eligible"], "candidate_count"]
    fit_artist_count = int(ledger.fitting_interactions["artist_id"].nunique())
    cold_catalog_per_user = len(ledger.primary_artist_universe) - fit_artist_count
    return {
        "method": result.method,
        "candidate_policy": result.candidate_policy,
        "ndcg_at_10": result.ndcg_at_10,
        "recall_at_10": result.recall_at_10,
        "coverage_at_10": result.coverage_at_10,
        "raw_evaluation_eligible_users": int(len(eligible)),
        "metric_eligible_users": len(result.eligible_user_ids),
        "users_without_retrievable_relevance": int((~eligible["metric_eligible"]).sum()),
        "heldout_pairs": int(eligible["heldout_count"].sum()),
        "heldout_cold_pairs": int(eligible["cold_heldout_count"].sum()),
        "candidate_universe_size": result.universe_size,
        "candidate_count_min": int(candidate_counts.min()),
        "candidate_count_max": int(candidate_counts.max()),
        "candidate_count_mean": float(candidate_counts.mean()),
        "cold_catalog_candidates": cold_catalog_per_user * len(result.eligible_user_ids),
        "distinct_recommended_artists": result.distinct_recommended_artists,
        "missing_representation_candidates": int(result.state_counts.get(MISSING_REPRESENTATION, 0)),
        "valid_zero_probability_candidates": int(result.state_counts.get(ZERO_PROBABILITY, 0)),
        "missing_component_candidates": int(result.state_counts.get(MISSING_COMPONENT, 0)),
        "invalid_numerical_candidates": int(result.state_counts.get(INVALID_NUMERICAL, 0)),
        "rankings_with_missing_scores": result.rankings_with_missing_scores,
        "failed_users": len(result.failed_user_ids),
        "runtime_seconds": result.runtime_seconds,
    }


def _evaluate_models(
    ledger: EvaluationPhaseLedger,
    policy: str,
    global_profile: GlobalPopularity,
    als_model: ALSModel,
    community_profile: CommunityPopularity,
    hybrid_profile: CommunityPopularity,
    membership: pd.DataFrame,
) -> dict[str, MethodEvaluation]:
    evaluations = {
        "global_popularity": evaluate_method(
            "global_popularity", policy, ledger,
            lambda _user, candidates: global_candidate_scores(global_profile, candidates),
        ),
        "als": evaluate_method(
            "als", policy, ledger,
            lambda user, candidates: als_candidate_scores(als_model, user, candidates),
        ),
        "community": evaluate_method(
            "community", policy, ledger,
            lambda user, candidates: community_candidate_scores(community_profile, membership, user, candidates),
        ),
        "hybrid": evaluate_method(
            "hybrid", policy, ledger,
            lambda user, candidates: hybrid_candidate_scores(
                als_candidate_scores(als_model, user, candidates),
                community_candidate_scores(hybrid_profile, membership, user, candidates),
                lambda_=SELECTED_HYBRID_LAMBDA,
            ),
        ),
    }
    for result in evaluations.values():
        if result.failed_user_ids:
            raise RuntimeError(f"{policy} {result.method} failed for users {result.failed_user_ids[:10]}")
        if result.eligible_user_ids != evaluations["global_popularity"].eligible_user_ids:
            raise RuntimeError(f"{policy} paired-user mismatch for {result.method}")
    return evaluations


def _git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        text=True, capture_output=True,
    )
    return completed.stdout.strip()


def _versions() -> dict[str, str]:
    packages = ("implicit", "networkx", "numpy", "pandas", "scipy", "pyarrow")
    return {name: importlib.metadata.version(name) for name in packages}


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_outputs(result: FinalAnalysis, evaluations: Mapping[str, MethodEvaluation]) -> None:
    output = result.output_directory
    output.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(output / "model_comparison.csv", index=False, float_format="%.12g")
    result.robustness.to_csv(output / "robustness.csv", index=False, float_format="%.12g")
    result.h1.anchors.to_csv(output / "h1_anchor_comparisons.csv", index=False, float_format="%.12g")
    result.h1.peer_reuse.to_csv(output / "h1_peer_reuse.csv", index=False)
    per_user = []
    for key, evaluation in evaluations.items():
        frame = evaluation.per_user.assign(result_key=key)
        per_user.append(frame[["result_key", "user_id", "ndcg_at_10", "recall_at_10"]])
    pd.concat(per_user, ignore_index=True).to_csv(
        output / "per_user_metrics.csv.gz", index=False, compression="gzip", float_format="%.12g"
    )
    summary = {
        "h2": asdict(result.h2),
        "h2_interpretation": result.h2_interpretation,
        "h1": {
            "anchor_count": len(result.h1.anchors),
            "candidate_anchor_counts": dict(result.h1.candidate_anchor_counts),
            "unique_peers": len(result.h1.peer_reuse),
            "maximum_peer_reuse": int(result.h1.peer_reuse["use_count"].max()),
            "mean_contrast": result.h1.mean_contrast,
            "jackknife_se": result.h1.jackknife_se,
            "ci_lower": result.h1.ci_lower,
            "ci_upper": result.h1.ci_upper,
            "jackknife_users": result.h1.jackknife_users,
            "jackknife_omitted": result.h1.jackknife_omitted,
        },
        "h3": dict(result.h3),
        "warnings": list(result.warnings),
        "runtime_seconds": result.runtime_seconds,
    }
    (output / "inference.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(result.provenance, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8"
    )


def run_final_analysis(
    archive_path: str | Path = "data/raw/hetrec2011-lastfm-2k.zip",
    p4_record_path: str | Path = ".agents/tasks/active/P4-validation.md",
    output_directory: str | Path = "data/processed/lastfm-v1/final",
) -> FinalAnalysis:
    """Execute the approved P5 refit, test evaluation, inference, and diagnostics."""
    started = perf_counter()
    archive_path = Path(archive_path)
    p4_record_path = Path(p4_record_path)
    output_directory = Path(output_directory)
    verify_selected_parameters(p4_record_path)

    data = load_hetrec(archive_path)
    split = split_interactions(data.interactions)
    sparse = compute_sparse_labels(split)
    community = detect_communities(data.interactions["user_id"], data.friendships)
    primary_ledger = build_phase_ledger(split, sparse, phase="test")
    validate_complete_sparse_labels(primary_ledger, sparse)
    if not sparse.labels["sparse"].astype(bool).all():
        raise RuntimeError("fixed H3 label state changed: expected every eligible user to be sparse")

    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        als_model = fit_implicit_als(primary_ledger.fitting_interactions, SELECTED_ALS)
    captured.extend(str(item.message) for item in caught)
    global_profile = fit_global_popularity(primary_ledger.fitting_interactions)
    community_profile = fit_community_popularity(
        primary_ledger.fitting_interactions, community.membership, beta=SELECTED_COMMUNITY_BETA
    )
    hybrid_profile = fit_community_popularity(
        primary_ledger.fitting_interactions, community.membership, beta=SELECTED_HYBRID_BETA
    )

    primary = _evaluate_models(
        primary_ledger, "primary_fit_artist_universe", global_profile, als_model,
        community_profile, hybrid_profile, community.membership,
    )
    h2 = paired_ndcg_bootstrap(primary["als"].per_user, primary["hybrid"].per_user)
    h2_interpretation = (
        "criterion_met: positive estimate and 95% percentile CI wholly above zero"
        if h2.estimate > 0.0 and h2.ci_lower > 0.0
        else "criterion_not_met: positive estimate with 95% percentile CI wholly above zero was not observed"
    )

    shuffled = shuffle_community_assignments(community.membership)
    shuffled_community_profile = fit_community_popularity(
        primary_ledger.fitting_interactions, shuffled, beta=SELECTED_COMMUNITY_BETA
    )
    shuffled_hybrid_profile = fit_community_popularity(
        primary_ledger.fitting_interactions, shuffled, beta=SELECTED_HYBRID_BETA
    )
    shuffled_results = {
        "shuffled_community": evaluate_method(
            "shuffled_community", "primary_fit_artist_universe", primary_ledger,
            lambda user, candidates: community_candidate_scores(shuffled_community_profile, shuffled, user, candidates),
        ),
        "shuffled_hybrid": evaluate_method(
            "shuffled_hybrid", "primary_fit_artist_universe", primary_ledger,
            lambda user, candidates: hybrid_candidate_scores(
                als_candidate_scores(als_model, user, candidates),
                community_candidate_scores(shuffled_hybrid_profile, shuffled, user, candidates),
                lambda_=SELECTED_HYBRID_LAMBDA,
            ),
        ),
    }
    for item in shuffled_results.values():
        if item.failed_user_ids or item.eligible_user_ids != primary["als"].eligible_user_ids:
            raise RuntimeError(f"negative-control integrity failure for {item.method}")

    catalog_ledger = full_catalog_ledger(primary_ledger, data.artists["artist_id"].astype(int).tolist())
    sensitivity = _evaluate_models(
        catalog_ledger, "full_catalog_sensitivity", global_profile, als_model,
        community_profile, hybrid_profile, community.membership,
    )
    h1 = run_h1_analysis(split.table, community.membership)

    comparison = pd.DataFrame([_result_row(primary[name], primary_ledger) for name in (
        "global_popularity", "als", "community", "hybrid"
    )])
    robustness_rows = [
        _result_row(shuffled_results["shuffled_community"], primary_ledger),
        _result_row(shuffled_results["shuffled_hybrid"], primary_ledger),
    ] + [_result_row(sensitivity[name], catalog_ledger) for name in (
        "global_popularity", "als", "community", "hybrid"
    )]
    robustness = pd.DataFrame(robustness_rows)
    h3 = {
        "estimable": False,
        "reason": "all eligible users are sparse under fixed lastfm-v1 labels; no non-sparse comparison group",
        "sparse_cutoff": sparse.cutoff,
        "raw_eligible_users": len(sparse.labels),
        "sparse_users": int(sparse.labels["sparse"].sum()),
        "non_sparse_users": int((~sparse.labels["sparse"].astype(bool)).sum()),
        "subgroup_interaction_contrast": None,
    }
    repository = Path(__file__).resolve().parents[2]
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "seed": SPLIT_SEED,
        "source_archive": str(archive_path),
        "selection_record": str(p4_record_path),
        "code_revision": _git_revision(repository),
        "working_tree_note": "revision identifies HEAD; allowed P2-P5 files may be uncommitted",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": _versions(),
        "thread_environment": {
            name: os.environ.get(name) for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
            )
        },
        "settings": {
            "als": asdict(SELECTED_ALS),
            "community_beta": SELECTED_COMMUNITY_BETA,
            "hybrid_beta": SELECTED_HYBRID_BETA,
            "hybrid_lambda": SELECTED_HYBRID_LAMBDA,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": SPLIT_SEED,
            "fit_scope": "train + validation + train_only_low_history",
            "louvain_seed": SPLIT_SEED,
            "louvain_communities": community.community_count,
            "louvain_modularity": community.modularity,
            "networkx_version": nx.__version__,
        },
    }
    all_evaluations = {
        **{f"primary_{key}": value for key, value in primary.items()},
        **shuffled_results,
        **{f"full_catalog_{key}": value for key, value in sensitivity.items()},
    }
    result = FinalAnalysis(
        comparison, robustness, h2, h2_interpretation, h1, h3, provenance,
        tuple(captured), perf_counter() - started, output_directory,
    )
    _write_outputs(result, all_evaluations)
    return result


def main() -> None:
    result = run_final_analysis()
    print(result.comparison.to_string(index=False))
    print(json.dumps({"h2": asdict(result.h2), "interpretation": result.h2_interpretation}, indent=2))
    print(f"outputs: {result.output_directory}")


if __name__ == "__main__":
    main()


__all__ = [
    "BOOTSTRAP_RESAMPLES", "SELECTED_ALS", "SELECTED_COMMUNITY_BETA", "SELECTED_HYBRID_BETA",
    "SELECTED_HYBRID_LAMBDA", "BootstrapResult", "FinalAnalysis", "H1Result", "MethodEvaluation",
    "descriptive_h1_jackknife", "evaluate_method", "full_catalog_ledger", "global_candidate_scores",
    "paired_ndcg_bootstrap", "run_final_analysis", "run_h1_analysis", "verify_selected_parameters",
]
