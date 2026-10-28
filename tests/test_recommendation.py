from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lastfm_project.evaluation_ledger import build_phase_ledger
from lastfm_project.recommendation import (
    INVALID_NUMERICAL,
    MISSING_REPRESENTATION,
    VALID,
    ZERO_PROBABILITY,
    CandidateScores,
    evaluate_ranker,
    hybrid_candidate_scores,
    ndcg_at_k,
    normalize_finite,
    rank_scores,
    recall_at_k,
)


def test_binary_metric_formulas_perfect_and_empty_rankings() -> None:
    assert ndcg_at_k([4, 2, 9], {4, 2}) == pytest.approx(1.0)
    assert recall_at_k([4, 2, 9], {4, 2}) == 1.0
    assert ndcg_at_k([], {4, 2}) == 0.0
    assert recall_at_k([], {4, 2}) == 0.0
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert ndcg_at_k([9, 4], {4, 2}) == pytest.approx(expected)


def test_score_ties_and_missing_sentinel_use_ascending_artist_id() -> None:
    scores = CandidateScores(
        (30, 10, 20, 40),
        np.array([1.0, 1.0, -np.inf, -np.inf]),
        (VALID, VALID, ZERO_PROBABILITY, MISSING_REPRESENTATION),
    )
    assert rank_scores(scores).artist_ids == (10, 30, 20, 40)


def test_z_normalization_population_sd_and_zero_variance() -> None:
    normalized = normalize_finite(CandidateScores((1, 2), np.array([1.0, 3.0]), (VALID, VALID)))
    assert normalized.scores.tolist() == [-1.0, 1.0]
    constant = normalize_finite(CandidateScores((1, 2), np.array([5.0, 5.0]), (VALID, VALID)))
    assert constant.scores.tolist() == [0.0, 0.0]


def test_hybrid_requires_both_finite_components_and_keeps_zero_probability_missing() -> None:
    mf = CandidateScores((1, 2, 3), np.array([1.0, 2.0, 3.0]), (VALID, VALID, VALID))
    community = CandidateScores(
        (1, 2, 3), np.array([4.0, -np.inf, 8.0]), (VALID, ZERO_PROBABILITY, VALID)
    )
    hybrid = hybrid_candidate_scores(mf, community, lambda_=0.5)
    assert hybrid.states[1] != VALID
    assert np.isneginf(hybrid.scores[1])
    assert np.isfinite(hybrid.scores[[0, 2]]).all()


def test_nonfinite_valid_score_fails_ranking_and_evaluation() -> None:
    invalid = CandidateScores((1,), np.array([np.nan]), (VALID,))
    ranking = rank_scores(invalid)
    assert ranking.failed and ranking.failure_reason == INVALID_NUMERICAL

    split = pd.DataFrame(
        [(1, 1, 1, "train"), (1, 2, 1, "validation")],
        columns=["user_id", "artist_id", "weight", "split"],
    ).astype({"user_id": "int64", "artist_id": "int64", "weight": "int64", "split": "string"})
    # Add another fitting user's artist so user 1 has a retrievable candidate.
    split.loc[len(split)] = [2, 2, 1, "train"]
    ledger = build_phase_ledger(split, phase="validation")
    result = evaluate_ranker(
        ledger,
        lambda user, candidates: CandidateScores(candidates, np.full(len(candidates), np.nan), (VALID,) * len(candidates)),
    )
    assert result.failed_user_ids == (1,)
    assert np.isnan(result.ndcg_at_10)


def test_evaluator_enforces_the_exact_shared_candidate_order() -> None:
    split = pd.DataFrame(
        [(1, 1, 1, "train"), (1, 2, 1, "validation"), (2, 2, 1, "train"), (2, 3, 1, "train")],
        columns=["user_id", "artist_id", "weight", "split"],
    ).astype({"user_id": "int64", "artist_id": "int64", "weight": "int64", "split": "string"})
    ledger = build_phase_ledger(split, phase="validation")
    result = evaluate_ranker(
        ledger,
        lambda user, candidates: CandidateScores(tuple(reversed(candidates)), np.zeros(len(candidates)), (VALID,) * len(candidates)),
    )
    assert result.failed_user_ids == (1,)

