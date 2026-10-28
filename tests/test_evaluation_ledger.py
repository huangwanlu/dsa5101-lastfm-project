from __future__ import annotations

import pandas as pd

from lastfm_project.evaluation_ledger import LEDGER_COLUMNS, build_evaluation_ledgers
from lastfm_project.splitting import compute_sparse_labels, split_interactions


def split_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Explicit split rows make the fit/held-out boundaries easy to audit.
    rows = [
        (1, 1, 1, "train"), (1, 2, 1, "train"), (1, 3, 1, "validation"), (1, 4, 1, "test"),
        (2, 2, 1, "train"), (2, 5, 1, "train"), (2, 4, 1, "validation"), (2, 6, 1, "test"),
        (3, 7, 1, "train_only_low_history"), (3, 8, 1, "train_only_low_history"),
        (4, 9, 1, "train"), (4, 10, 1, "train"), (4, 11, 1, "validation"), (4, 12, 1, "test"),
    ]
    split = pd.DataFrame(rows, columns=["user_id", "artist_id", "weight", "split"]).astype(
        {"user_id": "int64", "artist_id": "int64", "weight": "int64", "split": "string"}
    )
    sparse = pd.DataFrame({"user_id": [1, 2, 4], "sparse": pd.array([True, False, True], dtype="boolean")})
    return split, sparse


def test_fit_only_universes_history_exclusion_and_cold_items() -> None:
    split, sparse = split_fixture()
    ledgers = build_evaluation_ledgers(split, sparse)
    validation = ledgers.validation
    test = ledgers.test

    assert validation.primary_artist_universe == (1, 2, 5, 7, 8, 9, 10)
    assert test.primary_artist_universe == (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
    assert validation.candidate_artists(1) == (5, 7, 8, 9, 10)
    assert validation.relevance_artists(1) == ()  # artist 3 is cold in validation
    assert test.candidate_relevance(1) == ((4, 5, 7, 8, 9, 10, 11), (4,))

    v1 = validation.table.set_index("user_id").loc[1]
    assert bool(v1["raw_evaluation_eligible"]) is True
    assert bool(v1["metric_eligible"]) is False
    assert v1["cold_heldout_count"] == 1
    assert v1["candidate_count"] == 5
    assert validation.summary["excluded_users_due_to_cold_only_relevance"] == 3


def test_low_history_users_are_fit_only_and_sparse_labels_are_not_recomputed() -> None:
    split, sparse = split_fixture()
    ledgers = build_evaluation_ledgers(split.sample(frac=1, random_state=99), sparse)
    row = ledgers.validation.table.set_index("user_id").loc[3]
    assert bool(row["raw_evaluation_eligible"]) is False
    assert row["heldout_count"] == 0
    assert row["fit_history_count"] == 2
    assert row["sparse"] is pd.NA or pd.isna(row["sparse"])
    assert ledgers.validation.table.columns.tolist() == list(LEDGER_COLUMNS)
    assert ledgers.validation.table.equals(
        build_evaluation_ledgers(split, sparse).validation.table
    )


def test_real_split_sparse_labels_propagate_without_phase_redefinition() -> None:
    source = pd.DataFrame(
        [(user, artist, user * 100 + artist) for user, n in [(1, 7), (2, 8), (3, 10), (4, 6)] for artist in range(1, n + 1)],
        columns=["user_id", "artist_id", "weight"],
    ).astype("int64")
    split = split_interactions(source)
    sparse = compute_sparse_labels(split)
    ledgers = build_evaluation_ledgers(split, sparse)
    for phase in (ledgers.validation, ledgers.test):
        actual = phase.table.set_index("user_id")["sparse"].dropna().astype(bool).to_dict()
        expected = sparse.labels.set_index("user_id")["sparse"].astype(bool).to_dict()
        assert actual == expected
