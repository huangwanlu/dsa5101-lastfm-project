"""Deterministic validation/test eligibility ledgers for ``lastfm-v1``.

This module only derives fitting, candidate, and relevance sets.  It does not
fit models, score artists, or calculate recommendation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from .splitting import SPLIT_COLUMNS, SparseLabels, SplitResult


LEDGER_COLUMNS = (
    "user_id",
    "raw_evaluation_eligible",
    "metric_eligible",
    "split_train_history_count",
    "fit_history_count",
    "heldout_count",
    "retrievable_heldout_count",
    "cold_heldout_count",
    "candidate_count",
    "sparse",
)
PHASES = ("validation", "test")


def _table(value: SplitResult | pd.DataFrame) -> pd.DataFrame:
    result = value.table if isinstance(value, SplitResult) else value
    if tuple(result.columns) != SPLIT_COLUMNS:
        raise ValueError("split must have columns user_id, artist_id, weight, split")
    if result.duplicated(["user_id", "artist_id"]).any():
        raise ValueError("split contains duplicate user_id/artist_id pairs")
    if not result["split"].isin(("train", "validation", "test", "train_only_low_history")).all():
        raise ValueError("split contains an unknown partition")
    return result.copy(deep=True)


def _sparse_table(value: SparseLabels | pd.DataFrame | None) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame(columns=["user_id", "sparse"])
    result = value.labels if isinstance(value, SparseLabels) else value
    required = {"user_id", "sparse"}
    if not required.issubset(result.columns):
        raise ValueError("sparse labels must contain user_id and sparse")
    if result["user_id"].duplicated().any():
        raise ValueError("sparse labels contain duplicate user_id values")
    return result[["user_id", "sparse"]].copy(deep=True)


@dataclass(frozen=True)
class EvaluationPhaseLedger:
    """One phase's ledger and lazy set-reconstruction indexes."""

    phase: str
    table: pd.DataFrame
    fitting_interactions: pd.DataFrame
    primary_artist_universe: tuple[int, ...]
    _fit_artists_by_user: Mapping[int, frozenset[int]]
    _heldout_artists_by_user: Mapping[int, frozenset[int]]

    def candidate_artists(self, user_id: int) -> tuple[int, ...]:
        """Return the phase candidate set for ``user_id`` in ascending ID order."""
        excluded = self._fit_artists_by_user.get(int(user_id), frozenset())
        return tuple(artist for artist in self.primary_artist_universe if artist not in excluded)

    def relevance_artists(self, user_id: int) -> tuple[int, ...]:
        """Return held-out relevance intersected with that user's candidates."""
        candidates = set(self.candidate_artists(user_id))
        return tuple(sorted(self._heldout_artists_by_user.get(int(user_id), frozenset()) & candidates))

    def candidate_relevance(self, user_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return ``(candidate_artists, relevance_artists)`` for one user."""
        candidates = self.candidate_artists(user_id)
        relevance = tuple(sorted(self._heldout_artists_by_user.get(int(user_id), frozenset()) & set(candidates)))
        return candidates, relevance

    @property
    def summary(self) -> Mapping[str, object]:
        counts = self.table["candidate_count"]
        distribution = {
            int(count): int((counts == count).sum())
            for count in sorted(counts.unique())
        }
        return {
            "phase": self.phase,
            "raw_eligible_users": int(self.table["raw_evaluation_eligible"].sum()),
            "metric_eligible_users": int(self.table["metric_eligible"].sum()),
            "excluded_users_due_to_cold_only_relevance": int(
                (self.table["raw_evaluation_eligible"] & ~self.table["metric_eligible"] & (self.table["heldout_count"] > 0)).sum()
            ),
            "cold_heldout_pairs": int(self.table["cold_heldout_count"].sum()),
            "primary_artist_universe_size": len(self.primary_artist_universe),
            "candidate_count_distribution": distribution,
        }


@dataclass(frozen=True)
class EvaluationLedgers:
    """Validation and test ledgers derived from one frozen split."""

    validation: EvaluationPhaseLedger
    test: EvaluationPhaseLedger

    def __getitem__(self, phase: str) -> EvaluationPhaseLedger:
        if phase not in PHASES:
            raise KeyError(phase)
        return getattr(self, phase)


def _build_phase(table: pd.DataFrame, sparse: pd.DataFrame, phase: str) -> EvaluationPhaseLedger:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    fitting_splits = ("train", "train_only_low_history") if phase == "validation" else (
        "train", "validation", "train_only_low_history"
    )
    heldout_split = phase
    fitting = table.loc[table["split"].isin(fitting_splits), ["user_id", "artist_id", "weight", "split"]].copy()
    fitting = fitting.sort_values(["user_id", "artist_id"], kind="mergesort", ignore_index=True)
    universe = tuple(sorted(fitting["artist_id"].unique().tolist()))
    universe_set = set(universe)
    fit_by_user = {
        int(user_id): frozenset(group["artist_id"].astype(int))
        for user_id, group in fitting.groupby("user_id", sort=True)
    }
    heldout = table.loc[table["split"] == heldout_split, ["user_id", "artist_id"]]
    heldout_by_user = {
        int(user_id): frozenset(group["artist_id"].astype(int))
        for user_id, group in heldout.groupby("user_id", sort=True)
    }
    users = sorted(table["user_id"].astype(int).unique())
    rows = []
    for user_id in users:
        fit_artists = fit_by_user.get(user_id, frozenset())
        heldout_artists = heldout_by_user.get(user_id, frozenset())
        candidates = universe_set - set(fit_artists)
        retrievable = heldout_artists & candidates
        cold = heldout_artists - universe_set
        train_count = int(table.loc[(table["user_id"] == user_id) & (table["split"] == "train"), "artist_id"].nunique())
        rows.append((
            user_id, bool(user_id in heldout_by_user), bool(retrievable), train_count,
            len(fit_artists), len(heldout_artists), len(retrievable), len(cold), len(candidates),
        ))
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS[:-1])
    ledger = ledger.merge(sparse, on="user_id", how="left", sort=False)
    ledger["sparse"] = ledger["sparse"].astype("boolean")
    ledger = ledger[list(LEDGER_COLUMNS)].sort_values("user_id", kind="mergesort", ignore_index=True)
    ledger = ledger.astype({
        "user_id": "int64", "raw_evaluation_eligible": "bool", "metric_eligible": "bool",
        "split_train_history_count": "int64", "fit_history_count": "int64", "heldout_count": "int64",
        "retrievable_heldout_count": "int64", "cold_heldout_count": "int64", "candidate_count": "int64",
    })
    return EvaluationPhaseLedger(phase, ledger, fitting, universe, fit_by_user, heldout_by_user)


def build_phase_ledger(
    split: SplitResult | pd.DataFrame,
    sparse_labels: SparseLabels | pd.DataFrame | None = None,
    *,
    phase: str,
) -> EvaluationPhaseLedger:
    """Build one deterministic validation or test eligibility ledger."""
    return _build_phase(_table(split), _sparse_table(sparse_labels), phase)


def build_evaluation_ledgers(
    split: SplitResult | pd.DataFrame,
    sparse_labels: SparseLabels | pd.DataFrame | None = None,
) -> EvaluationLedgers:
    """Build both phase ledgers using the same frozen sparse-label table."""
    source = _table(split)
    sparse = _sparse_table(sparse_labels)
    return EvaluationLedgers(_build_phase(source, sparse, "validation"), _build_phase(source, sparse, "test"))


__all__ = [
    "LEDGER_COLUMNS", "PHASES", "EvaluationPhaseLedger", "EvaluationLedgers",
    "build_phase_ledger", "build_evaluation_ledgers",
]
