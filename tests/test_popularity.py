from __future__ import annotations

from math import log1p

import pandas as pd
import pytest

from lastfm_project.communities import detect_communities
from lastfm_project.evaluation_ledger import build_phase_ledger
from lastfm_project.popularity import fit_community_popularity, fit_global_popularity, validate_complete_sparse_labels


def interactions(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "artist_id", "weight"]).astype("int64")


def test_global_popularity_uses_log_activity_and_stable_ties() -> None:
    profile = fit_global_popularity(interactions([(1, 10, 3), (2, 20, 1), (3, 20, 1), (4, 30, 3)]))
    assert profile.scores[10] == pytest.approx(log1p(3))
    assert profile.scores[20] == pytest.approx(2 * log1p(1))
    assert profile.score_candidates([30, 10, 20]).artist_id.tolist() == [10, 20, 30]
    assert profile.score_candidates([40, 50]).artist_id.tolist() == [40, 50]


def test_community_shrinkage_formula_and_empty_community_backoff() -> None:
    fit = interactions([(1, 10, 3), (2, 20, 1)])
    membership = pd.DataFrame({"user_id": [1, 2, 3], "community_id": [1, 1, 3]}).astype("int64")
    profile = fit_community_popularity(fit, membership, beta=10)
    q10 = log1p(3) / (log1p(3) + log1p(1))
    expected = (log1p(3) + 10 * q10) / (log1p(3) + log1p(1) + 10)
    assert profile.scores_by_community[1][10] == pytest.approx(expected)
    assert profile.scores_for_user(3, membership) == profile.global_scores
    assert sum(profile.scores_by_community[3].values()) == pytest.approx(1.0)


def test_candidate_exclusion_and_tie_breaking_are_external_and_stable() -> None:
    profile = fit_global_popularity(interactions([(1, 10, 1), (1, 11, 1)]))
    # The phase ledger supplies the fitting-only candidate set; the profile does not add history back.
    split = interactions([(1, 10, 1), (1, 11, 1), (1, 12, 1), (1, 13, 1), (1, 14, 1), (1, 15, 1), (1, 16, 1)])
    split["split"] = pd.Series(["train"] * 5 + ["validation", "test"], dtype="string")
    ledger = build_phase_ledger(split, phase="validation")
    candidates = ledger.candidate_artists(1)
    assert 10 not in candidates and 11 not in candidates
    assert profile.score_candidates(candidates).artist_id.tolist() == sorted(candidates)


def test_missing_fixed_sparse_label_is_rejected_for_evaluation() -> None:
    split = interactions([(1, i, 1) for i in range(1, 8)] + [(2, i, 1) for i in range(1, 8)])
    split["split"] = pd.Series(["train"] * 5 + ["validation", "test"] + ["train"] * 5 + ["validation", "test"], dtype="string")
    ledger = build_phase_ledger(split, phase="validation")
    labels = pd.DataFrame({"user_id": [1], "sparse": pd.array([True], dtype="boolean")})
    with pytest.raises(ValueError, match="missing fixed sparse labels"):
        validate_complete_sparse_labels(ledger, labels)


def test_missing_or_duplicate_fitting_user_membership_is_rejected() -> None:
    fit = interactions([(1, 10, 1), (2, 20, 1)])
    missing = pd.DataFrame({"user_id": [1], "community_id": [1]}).astype("int64")
    with pytest.raises(ValueError, match="missing community assignments.*2"):
        fit_community_popularity(fit, missing, beta=10)
    duplicate = pd.DataFrame({"user_id": [1, 1, 2], "community_id": [1, 2, 2]}).astype("int64")
    with pytest.raises(ValueError, match="duplicate users"):
        fit_community_popularity(fit, duplicate, beta=10)
