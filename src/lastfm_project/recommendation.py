"""Shared scoring, ranking, and binary top-10 metrics for ``lastfm-v1``."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .als import ALSModel
from .evaluation_ledger import EvaluationPhaseLedger
from .popularity import CommunityPopularity


VALID = "valid"
MISSING_REPRESENTATION = "missing_representation"
ZERO_PROBABILITY = "valid_zero_probability"
MISSING_COMPONENT = "missing_component"
INVALID_NUMERICAL = "invalid_numerical_result"
_EXPECTED_NONFINITE_STATES = {MISSING_REPRESENTATION, ZERO_PROBABILITY, MISSING_COMPONENT}


@dataclass(frozen=True)
class CandidateScores:
    artist_ids: tuple[int, ...]
    scores: np.ndarray
    states: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.artist_ids) != len(self.scores) or len(self.scores) != len(self.states):
            raise ValueError("candidate score fields must have equal lengths")
        if len(set(self.artist_ids)) != len(self.artist_ids):
            raise ValueError("candidate artist IDs must be unique")


@dataclass(frozen=True)
class Ranking:
    table: pd.DataFrame
    failed: bool
    failure_reason: str | None
    state_counts: Mapping[str, int]

    @property
    def artist_ids(self) -> tuple[int, ...]:
        return tuple(self.table["artist_id"].astype(int))


@dataclass(frozen=True)
class MetricEvaluation:
    ndcg_at_10: float
    recall_at_10: float
    per_user: pd.DataFrame
    eligible_user_ids: tuple[int, ...]
    failed_user_ids: tuple[int, ...]
    state_counts: Mapping[str, int]

    @property
    def successful(self) -> bool:
        return not self.failed_user_ids and bool(self.eligible_user_ids)


def ndcg_at_k(ranking: tuple[int, ...] | list[int], relevance: set[int] | tuple[int, ...], *, k: int = 10) -> float:
    relevant = set(int(value) for value in relevance)
    if not relevant:
        raise ValueError("NDCG is undefined for empty relevance")
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, artist_id in enumerate(ranking[:k], start=1)
        if int(artist_id) in relevant
    )
    ideal_length = min(k, len(relevant))
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_length + 1))
    return float(dcg / idcg)


def recall_at_k(ranking: tuple[int, ...] | list[int], relevance: set[int] | tuple[int, ...], *, k: int = 10) -> float:
    relevant = set(int(value) for value in relevance)
    if not relevant:
        raise ValueError("recall is undefined for empty relevance")
    return float(len(set(int(value) for value in ranking[:k]) & relevant) / len(relevant))


def normalize_finite(component: CandidateScores) -> CandidateScores:
    """Per-user population z-normalization over finite valid scores only."""
    values = np.asarray(component.scores, dtype=np.float64)
    states = np.asarray(component.states, dtype=object)
    valid = states == VALID
    if np.any(valid & ~np.isfinite(values)):
        raise FloatingPointError("a valid component score is nonfinite")
    unexpected_finite = (~valid) & np.isfinite(values)
    if unexpected_finite.any():
        raise FloatingPointError("a non-valid score state contains a finite score")
    unknown_nonfinite = (~valid) & ~np.isin(states, tuple(_EXPECTED_NONFINITE_STATES))
    if unknown_nonfinite.any() or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise FloatingPointError("invalid numerical component output")
    if not valid.any():
        raise ValueError("component has no finite candidate scores")
    finite = values[valid]
    mean = float(finite.mean())
    standard_deviation = float(finite.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(standard_deviation):
        raise FloatingPointError("nonfinite component normalization statistic")
    normalized = np.full(len(values), -np.inf, dtype=np.float64)
    normalized[valid] = 0.0 if standard_deviation == 0.0 else (finite - mean) / standard_deviation
    if not np.isfinite(normalized[valid]).all():
        raise FloatingPointError("nonfinite normalized component score")
    return CandidateScores(component.artist_ids, normalized, component.states)


def als_candidate_scores(model: ALSModel, user_id: int, candidate_artists: tuple[int, ...]) -> CandidateScores:
    scores, represented = model.score(user_id, candidate_artists)
    states = tuple(VALID if present else MISSING_REPRESENTATION for present in represented)
    return CandidateScores(tuple(candidate_artists), scores, states)


def community_candidate_scores(
    profile: CommunityPopularity,
    membership: pd.DataFrame,
    user_id: int,
    candidate_artists: tuple[int, ...],
) -> CandidateScores:
    probabilities = profile.scores_for_user(user_id, membership)
    values = np.empty(len(candidate_artists), dtype=np.float64)
    states: list[str] = []
    for index, artist_id in enumerate(candidate_artists):
        probability = float(probabilities.get(int(artist_id), 0.0))
        if not np.isfinite(probability) or probability < 0.0:
            values[index] = np.nan
            states.append(INVALID_NUMERICAL)
        elif probability == 0.0:
            values[index] = -np.inf
            states.append(ZERO_PROBABILITY)
        else:
            values[index] = np.log(probability)
            states.append(VALID)
    return CandidateScores(tuple(candidate_artists), values, tuple(states))


def hybrid_candidate_scores(
    mf: CandidateScores, community: CandidateScores, *, lambda_: float
) -> CandidateScores:
    if lambda_ not in (0.25, 0.50, 0.75):
        raise ValueError("lambda must be one of 0.25, 0.50, 0.75")
    if mf.artist_ids != community.artist_ids:
        raise ValueError("hybrid components must use identical ordered candidates")
    mf_z = normalize_finite(mf)
    community_z = normalize_finite(community)
    mf_valid = np.asarray(mf_z.states) == VALID
    community_valid = np.asarray(community_z.states) == VALID
    shared = mf_valid & community_valid
    if not shared.any():
        raise ValueError("hybrid has no candidates with both finite components")
    scores = np.full(len(mf.artist_ids), -np.inf, dtype=np.float64)
    scores[shared] = lambda_ * mf_z.scores[shared] + (1.0 - lambda_) * community_z.scores[shared]
    if not np.isfinite(scores[shared]).all():
        raise FloatingPointError("nonfinite hybrid score")
    states = tuple(VALID if available else MISSING_COMPONENT for available in shared)
    return CandidateScores(mf.artist_ids, scores, states)


def rank_scores(component: CandidateScores, *, limit: int | None = None) -> Ranking:
    values = np.asarray(component.scores, dtype=np.float64)
    states = np.asarray(component.states, dtype=object)
    invalid = (states == INVALID_NUMERICAL) | np.isnan(values) | np.isposinf(values)
    invalid |= (states == VALID) & ~np.isfinite(values)
    invalid |= (states != VALID) & np.isfinite(values)
    invalid |= (~np.isin(states, (VALID, *_EXPECTED_NONFINITE_STATES)))
    counts = {str(state): int((states == state).sum()) for state in sorted(set(component.states))}
    if invalid.any():
        return Ranking(
            pd.DataFrame(columns=["artist_id", "score", "state"]),
            True,
            INVALID_NUMERICAL,
            counts,
        )
    artist_ids = np.asarray(component.artist_ids, dtype=np.int64)
    missing = (states != VALID).astype(np.int8)
    sort_scores = np.where(states == VALID, -values, 0.0)
    order = np.lexsort((artist_ids, sort_scores, missing))
    if limit is not None:
        if limit < 0:
            raise ValueError("ranking limit must be nonnegative")
        order = order[:limit]
    table = pd.DataFrame(
        {"artist_id": artist_ids[order], "score": values[order], "state": states[order]}
    ).astype(
        {"artist_id": "int64", "score": "float64", "state": "string"}
    )
    return Ranking(table, False, None, counts)


def evaluate_ranker(
    ledger: EvaluationPhaseLedger,
    scorer: Callable[[int, tuple[int, ...]], CandidateScores],
    *,
    k: int = 10,
) -> MetricEvaluation:
    """Evaluate exactly the ledger's shared candidates and metric-eligible users."""
    users = tuple(
        ledger.table.loc[ledger.table["metric_eligible"], "user_id"].astype(int)
    )
    rows: list[tuple[int, float, float]] = []
    failed: list[int] = []
    aggregate_counts: dict[str, int] = {}
    for user_id in users:
        candidates, relevance = ledger.candidate_relevance(user_id)
        try:
            component = scorer(user_id, candidates)
            if component.artist_ids != candidates:
                raise ValueError("scorer did not preserve the ledger candidate set and order")
            ranking = rank_scores(component, limit=k)
        except (FloatingPointError, ValueError, KeyError):
            failed.append(user_id)
            continue
        for state, count in ranking.state_counts.items():
            aggregate_counts[state] = aggregate_counts.get(state, 0) + count
        if ranking.failed:
            failed.append(user_id)
            continue
        ranked_ids = ranking.artist_ids
        rows.append((user_id, ndcg_at_k(ranked_ids, relevance, k=k), recall_at_k(ranked_ids, relevance, k=k)))
    per_user = pd.DataFrame(rows, columns=["user_id", "ndcg_at_10", "recall_at_10"]).astype(
        {"user_id": "int64", "ndcg_at_10": "float64", "recall_at_10": "float64"}
    )
    if failed or not rows:
        macro_ndcg = macro_recall = float("nan")
    else:
        macro_ndcg = float(per_user["ndcg_at_10"].mean())
        macro_recall = float(per_user["recall_at_10"].mean())
    return MetricEvaluation(
        macro_ndcg, macro_recall, per_user, users, tuple(failed), aggregate_counts
    )


__all__ = [
    "INVALID_NUMERICAL", "MISSING_COMPONENT", "MISSING_REPRESENTATION", "VALID",
    "ZERO_PROBABILITY", "CandidateScores", "MetricEvaluation", "Ranking",
    "als_candidate_scores", "community_candidate_scores", "evaluate_ranker",
    "hybrid_candidate_scores", "ndcg_at_k", "normalize_finite", "rank_scores", "recall_at_k",
]
