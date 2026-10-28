from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lastfm_project.als import ALSConfig, ALSModel
from lastfm_project.evaluation_ledger import build_phase_ledger
from lastfm_project.final_evaluation import (
    descriptive_h1_jackknife,
    evaluate_method,
    full_catalog_ledger,
    paired_ndcg_bootstrap,
    run_h1_analysis,
    verify_frozen_selection,
)
from lastfm_project.recommendation import MISSING_REPRESENTATION, VALID, CandidateScores, als_candidate_scores


def _split(rows: list[tuple[int, int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "artist_id", "weight", "split"]).astype(
        {"user_id": "int64", "artist_id": "int64", "weight": "int64", "split": "string"}
    )


def test_paired_bootstrap_is_reproducible_and_preserves_pairing() -> None:
    als = pd.DataFrame({"user_id": [1, 2, 3], "ndcg_at_10": [0.1, 0.4, 0.3]})
    hybrid = pd.DataFrame({"user_id": [1, 2, 3], "ndcg_at_10": [0.2, 0.2, 0.6]})
    first = paired_ndcg_bootstrap(als, hybrid)
    second = paired_ndcg_bootstrap(als, hybrid)
    assert first == second
    assert first.estimate == pytest.approx((0.1 - 0.2 + 0.3) / 3)
    with pytest.raises(ValueError, match="paired-user mismatch"):
        paired_ndcg_bootstrap(als, hybrid.iloc[::-1].reset_index(drop=True))


def test_frozen_selection_record_must_match_all_selected_parameters(tmp_path) -> None:
    valid = tmp_path / "P4.md"
    valid.write_text(
        "ALS: factors 32, regularization 0.10, alpha 1, iterations 20, seed 5101.\n"
        "Community: beta 1,000.\n"
        "Hybrid: selected ALS plus beta 100 and lambda 0.25.\n",
        encoding="utf-8",
    )
    verify_frozen_selection(valid)
    valid.write_text(valid.read_text(encoding="utf-8").replace("lambda 0.25", "lambda 0.50"), encoding="utf-8")
    with pytest.raises(ValueError, match="Hybrid"):
        verify_frozen_selection(valid)


def test_h1_peer_selection_activity_matching_reuse_and_jackknife() -> None:
    rows = []
    artist_sets = {
        1: range(1, 6), 2: range(1, 6), 3: range(6, 11),
        4: range(6, 11), 5: range(11, 16), 6: range(11, 16),
    }
    for user, artists in artist_sets.items():
        rows.extend((user, artist, 1, "train") for artist in artists)
    membership = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6], "community_id": [1, 1, 3, 3, 5, 5]
    }).astype("int64")
    result = run_h1_analysis(_split(rows), membership)
    again = run_h1_analysis(_split(list(reversed(rows))), membership.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(result.anchors, again.anchors)
    assert result.candidate_anchor_counts["5-10"] == 6
    assert len(result.anchors) == 6
    assert int(result.peer_reuse["use_count"].sum()) == 12
    assert np.isfinite(result.jackknife_se)


def test_jackknife_removes_every_row_where_user_participates() -> None:
    anchors = pd.DataFrame({
        "anchor_user_id": list(range(1, 9)),
        "within_peer_id": [2, 3, 4, 5, 6, 7, 8, 1],
        "cross_peer_id": [4, 5, 6, 7, 8, 1, 2, 3],
        "contrast": [0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.15, 0.05],
    })
    se, lower, upper, users, omitted = descriptive_h1_jackknife(anchors)
    assert users == 8 and omitted == 0
    assert se >= 0 and lower <= anchors["contrast"].mean() <= upper


def test_coverage_and_missing_score_accounting() -> None:
    ledger = build_phase_ledger(_split([
        (1, 1, 1, "train"), (1, 2, 1, "test"),
        (2, 2, 1, "train"), (2, 1, 1, "test"), (2, 3, 1, "train"),
    ]), phase="test")

    def scorer(_user: int, candidates: tuple[int, ...]) -> CandidateScores:
        states = tuple(VALID if artist != 3 else MISSING_REPRESENTATION for artist in candidates)
        scores = np.array([float(-artist) if state == VALID else -np.inf for artist, state in zip(candidates, states)])
        return CandidateScores(candidates, scores, states)

    result = evaluate_method("synthetic", "primary", ledger, scorer, k=1)
    assert result.failed_user_ids == ()
    assert result.coverage_at_10 == pytest.approx(2 / 3)
    assert result.state_counts[MISSING_REPRESENTATION] == 1
    assert result.rankings_with_missing_scores == 1


def test_full_catalog_sensitivity_uses_identical_candidates_and_counts_cold_items() -> None:
    primary = build_phase_ledger(_split([
        (1, 1, 1, "train"), (1, 4, 1, "test"),
        (2, 2, 1, "train"), (2, 1, 1, "test"),
    ]), phase="test")
    full = full_catalog_ledger(primary, [1, 2, 3, 4])
    assert full.candidate_artists(1) == (2, 3, 4)
    assert full.relevance_artists(1) == (4,)
    row = full.table.set_index("user_id").loc[1]
    assert row["cold_heldout_count"] == 1
    assert bool(row["metric_eligible"])

    model = ALSModel(
        ALSConfig(32, 0.1, 1.0), (1, 2), (1, 2), {1: 0, 2: 1}, {1: 0, 2: 1},
        np.ones((2, 32)), np.ones((2, 32)),
    )
    result = evaluate_method(
        "als", "full", full,
        lambda user, candidates: als_candidate_scores(model, user, candidates),
    )
    assert result.state_counts[MISSING_REPRESENTATION] == 4
    assert result.rankings_with_missing_scores == 2


def test_numerical_failure_is_not_silently_dropped() -> None:
    ledger = build_phase_ledger(_split([
        (1, 1, 1, "train"), (1, 2, 1, "test"), (2, 2, 1, "train"),
    ]), phase="test")
    result = evaluate_method(
        "bad", "primary", ledger,
        lambda _user, candidates: CandidateScores(candidates, np.full(len(candidates), np.nan), (VALID,) * len(candidates)),
    )
    assert result.failed_user_ids == (1,)
    assert np.isnan(result.ndcg_at_10)
