from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from lastfm_project.splitting import compute_sparse_labels, split_interactions


def interactions_for_counts(counts: dict[int, int]) -> pd.DataFrame:
    rows = [
        (user_id, artist_id, user_id * 1000 + artist_id)
        for user_id, count in counts.items()
        for artist_id in range(1, count + 1)
    ]
    return pd.DataFrame(rows, columns=["user_id", "artist_id", "weight"]).astype("int64")


def expected_rank(user_id: int, artist_ids: list[int]) -> list[int]:
    return sorted(
        artist_ids,
        key=lambda artist_id: (
            hashlib.sha256(f"5101|{user_id}|{artist_id}".encode("ascii")).digest(),
            artist_id,
        ),
    )


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, {"train_only_low_history": 1}), (6, {"train_only_low_history": 6}), (7, {"train": 5, "validation": 1, "test": 1}), (8, {"train": 6, "validation": 1, "test": 1}), (10, {"train": 7, "validation": 2, "test": 1})],
)
def test_boundary_counts_and_nonempty_eligible_partitions(count: int, expected: dict[str, int]) -> None:
    result = split_interactions(interactions_for_counts({1: count}))
    assert result.table["split"].value_counts().to_dict() == expected
    assert result.table.columns.tolist() == ["user_id", "artist_id", "weight", "split"]
    assert result.table[["user_id", "artist_id", "weight"]].dtypes.astype(str).tolist() == ["int64"] * 3


def test_hash_ranking_and_assignment_are_reproducible() -> None:
    source = interactions_for_counts({3: 10})
    result = split_interactions(source)
    ranked = expected_rank(3, list(range(1, 11)))
    expected = dict(zip(ranked, ["train"] * 7 + ["validation"] * 2 + ["test"]))
    actual = dict(zip(result.table["artist_id"], result.table["split"]))
    assert actual == expected
    assert result.metadata["seed"] == 5101
    assert result.metadata["protocol_id"] == "lastfm-v1"


def test_shuffled_input_is_identical_and_input_is_unmodified() -> None:
    source = interactions_for_counts({1: 7, 2: 8, 3: 6})
    original = source.copy(deep=True)
    first = split_interactions(source).table
    shuffled = split_interactions(source.sample(frac=1, random_state=123).reset_index(drop=True)).table
    pd.testing.assert_frame_equal(first, shuffled)
    pd.testing.assert_frame_equal(source, original)


def test_pair_conservation_weights_and_summary() -> None:
    source = interactions_for_counts({1: 7, 2: 6, 3: 10})
    result = split_interactions(source)
    actual = result.table.set_index(["user_id", "artist_id"])["weight"].sort_index()
    expected = source.set_index(["user_id", "artist_id"])["weight"].sort_index()
    pd.testing.assert_series_equal(actual, expected)
    assert result.table.duplicated(["user_id", "artist_id"]).sum() == 0
    assert result.summary["train"]["rows"] == 5 + 7
    assert result.summary["train_only_low_history"] == {"rows": 6, "users": 1}


def test_largest_remainder_tie_break_is_train_then_validation_then_test() -> None:
    # n=7 has quotas 4.9, 1.05, 1.05: both one-item remainders tie.
    result = split_interactions(interactions_for_counts({1: 7}))
    assert result.table["split"].value_counts().to_dict() == {"train": 5, "validation": 1, "test": 1}


def test_sparse_labels_use_nearest_rank_and_include_cutoff_ties() -> None:
    # Eligible train counts are 5, 5, 6, 7; nearest-rank Q1 is 5.
    source = interactions_for_counts({1: 7, 2: 7, 3: 8, 4: 10, 5: 6})
    split = split_interactions(source)
    sparse = compute_sparse_labels(split)
    assert sparse.cutoff == 5
    assert sparse.metadata["eligible_user_count"] == 4
    assert sparse.labels.loc[sparse.labels["sparse"], "user_id"].tolist() == [1, 2]
    assert set(sparse.labels["user_id"]) == {1, 2, 3, 4}
    assert 5 not in set(sparse.labels["user_id"])


def test_sparse_labels_ignore_low_history_and_held_out_rows() -> None:
    source = interactions_for_counts({1: 7, 2: 6})
    split = split_interactions(source)
    sparse = compute_sparse_labels(split.table)
    assert sparse.labels["user_id"].tolist() == [1]
    assert sparse.labels["train_artist_count"].tolist() == [5]


def test_invalid_source_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="dtype int64"):
        split_interactions(
            pd.DataFrame({"user_id": [1], "artist_id": [1], "weight": [2]}).astype("int32")
        )
