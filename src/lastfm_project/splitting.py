"""Deterministic interaction splitting for the fixed ``lastfm-v1`` protocol."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil, floor
from typing import Mapping

import pandas as pd


PROTOCOL_ID = "lastfm-v1"
SPLIT_SEED = 5101
ELIGIBILITY_THRESHOLD = 7
SPLIT_COLUMNS = ("user_id", "artist_id", "weight", "split")
_PARTITIONS = ("train", "validation", "test")
_LOW_HISTORY = "train_only_low_history"
_PROPORTIONS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class SplitResult:
    """A validated split table with its audit summary and provenance metadata."""

    table: pd.DataFrame
    summary: Mapping[str, Mapping[str, int]]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class SparseLabels:
    """Fixed sparse-history labels and the nearest-rank cutoff."""

    labels: pd.DataFrame
    cutoff: int
    metadata: Mapping[str, object]


def _allocation_counts(n: int) -> tuple[int, int, int]:
    quotas = [proportion * n for proportion in _PROPORTIONS]
    counts = [floor(quota) for quota in quotas]
    remaining = n - sum(counts)
    order = sorted(
        range(len(_PARTITIONS)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    result = tuple(counts)
    if min(result) < 1 or sum(result) != n:
        raise AssertionError(f"invalid eligible allocation for n={n}: {result}")
    return result  # type: ignore[return-value]


def _rank_user_rows(user_id: int, rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(
        rows,
        key=lambda row: (
            sha256(f"{SPLIT_SEED}|{user_id}|{row[0]}".encode("ascii")).digest(),
            row[0],
        ),
    )


def _validate_source(interactions: pd.DataFrame) -> None:
    if tuple(interactions.columns) != ("user_id", "artist_id", "weight"):
        raise ValueError("interactions must have columns user_id, artist_id, weight")
    if any(str(interactions[column].dtype) != "int64" for column in interactions.columns):
        raise ValueError("interactions integer columns must have dtype int64")
    if interactions.duplicated(["user_id", "artist_id"]).any():
        raise ValueError("interactions contain duplicate user_id/artist_id pairs")


def _validate_result(source: pd.DataFrame, result: pd.DataFrame) -> None:
    if tuple(result.columns) != SPLIT_COLUMNS:
        raise AssertionError("split output columns changed")
    if any(str(result[column].dtype) != "int64" for column in ("user_id", "artist_id", "weight")):
        raise AssertionError("split integer columns are not int64")
    if result.duplicated(["user_id", "artist_id"]).any():
        raise AssertionError("split output contains duplicate pairs")
    source_pairs = source.set_index(["user_id", "artist_id"])["weight"].sort_index()
    result_pairs = result.set_index(["user_id", "artist_id"])["weight"].sort_index()
    if not source_pairs.equals(result_pairs):
        raise AssertionError("split output does not conserve source pairs and weights")
    if not result["split"].isin(_PARTITIONS + (_LOW_HISTORY,)).all():
        raise AssertionError("split output contains an unknown partition")
    for user_id, group in result.groupby("user_id", sort=False):
        n = len(group)
        counts = group["split"].value_counts()
        if n < ELIGIBILITY_THRESHOLD:
            if set(counts.index) != {_LOW_HISTORY}:
                raise AssertionError(f"low-history user {user_id} has held-out rows")
        else:
            expected = dict(zip(_PARTITIONS, _allocation_counts(n)))
            if {partition: int(counts.get(partition, 0)) for partition in _PARTITIONS} != expected:
                raise AssertionError(f"user {user_id} has incorrect allocation")


def split_interactions(interactions: pd.DataFrame) -> SplitResult:
    """Create the deterministic fixed 70/15/15 interaction split.

    ``interactions`` must be the validated P2.1 table. The input is never
    modified; every source pair and weight appears exactly once in ``table``.
    """
    _validate_source(interactions)
    source = interactions.copy(deep=True)
    assignments: list[tuple[int, int, int, str]] = []
    for user_id, group in source.groupby("user_id", sort=True):
        rows = [(int(row.artist_id), int(row.weight)) for row in group.itertuples(index=False)]
        ranked = _rank_user_rows(int(user_id), rows)
        if len(ranked) < ELIGIBILITY_THRESHOLD:
            assignments.extend((int(user_id), artist_id, weight, _LOW_HISTORY) for artist_id, weight in ranked)
            continue
        counts = _allocation_counts(len(ranked))
        start = 0
        for partition, count in zip(_PARTITIONS, counts):
            assignments.extend(
                (int(user_id), artist_id, weight, partition)
                for artist_id, weight in ranked[start : start + count]
            )
            start += count
    table = pd.DataFrame(assignments, columns=SPLIT_COLUMNS)
    table = table.sort_values(["user_id", "artist_id"], kind="mergesort", ignore_index=True)
    table = table.astype({"user_id": "int64", "artist_id": "int64", "weight": "int64", "split": "string"})
    _validate_result(source, table)
    summary = {
        partition: {
            "rows": int((table["split"] == partition).sum()),
            "users": int(table.loc[table["split"] == partition, "user_id"].nunique()),
        }
        for partition in _PARTITIONS + (_LOW_HISTORY,)
    }
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "seed": SPLIT_SEED,
        "method": "per-user SHA-256 ranking with largest-remainder apportionment",
        "eligibility_threshold": ELIGIBILITY_THRESHOLD,
        "proportions": _PROPORTIONS,
        "tie_break": "train, validation, test",
    }
    return SplitResult(table=table, summary=summary, metadata=metadata)


def compute_sparse_labels(split: SplitResult | pd.DataFrame) -> SparseLabels:
    """Fix sparse labels from eligible users' split-train artist counts only."""
    table = split.table if isinstance(split, SplitResult) else split
    if tuple(table.columns) != SPLIT_COLUMNS:
        raise ValueError("split must have columns user_id, artist_id, weight, split")
    counts = (
        table.loc[table["split"] == "train"]
        .groupby("user_id")["artist_id"]
        .nunique()
        .astype("int64")
    )
    eligible = (
        table.loc[table["split"].isin(_PARTITIONS)]
        .groupby("user_id")
        .size()
        .loc[lambda values: values >= ELIGIBILITY_THRESHOLD]
        .index
    )
    labels = pd.DataFrame({"user_id": sorted(eligible)}, dtype="int64")
    labels["train_artist_count"] = labels["user_id"].map(counts).astype("int64")
    if labels.empty:
        raise ValueError("cannot compute sparse cutoff without eligible users")
    cutoff_rank = ceil(0.25 * len(labels))
    cutoff = int(labels["train_artist_count"].sort_values(ignore_index=True).iloc[cutoff_rank - 1])
    labels["sparse"] = labels["train_artist_count"] <= cutoff
    labels = labels.astype({"user_id": "int64", "train_artist_count": "int64", "sparse": "boolean"})
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "basis": "split-train distinct artist counts",
        "eligible_user_count": len(labels),
        "nearest_rank": cutoff_rank,
    }
    return SparseLabels(labels=labels, cutoff=cutoff, metadata=metadata)


__all__ = ["PROTOCOL_ID", "SPLIT_SEED", "ELIGIBILITY_THRESHOLD", "SplitResult", "SparseLabels", "split_interactions", "compute_sparse_labels"]
