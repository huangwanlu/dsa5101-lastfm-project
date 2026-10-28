"""Fit-only global and community popularity recommenders."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Iterable, Mapping

import pandas as pd

from .communities import MEMBERSHIP_COLUMNS
from .evaluation_ledger import EvaluationPhaseLedger
from .splitting import SparseLabels

BETA_VALUES = (10.0, 100.0, 1000.0)


def _fit_table(fitting_interactions: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "artist_id", "weight"}
    if not required.issubset(fitting_interactions.columns):
        raise ValueError("fitting interactions must contain user_id, artist_id, weight")
    result = fitting_interactions[["user_id", "artist_id", "weight"]].copy()
    if (result["weight"] <= 0).any():
        raise ValueError("fitting interaction weights must be positive")
    if result.duplicated(["user_id", "artist_id"]).any():
        raise ValueError("fitting interactions contain duplicate pairs")
    result["activity"] = result["weight"].map(log1p)
    return result


def _ranked(scores: Mapping[int, float], candidate_artists: Iterable[int]) -> pd.DataFrame:
    rows = [(int(artist_id), float(scores.get(int(artist_id), 0.0))) for artist_id in set(candidate_artists)]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return pd.DataFrame(rows, columns=["artist_id", "score"]).astype({"artist_id": "int64", "score": "float64"})


def validate_complete_sparse_labels(
    ledger: EvaluationPhaseLedger, sparse_labels: SparseLabels | pd.DataFrame
) -> None:
    """Require exactly one frozen sparse label for every raw-eligible ledger user."""
    labels = sparse_labels.labels if isinstance(sparse_labels, SparseLabels) else sparse_labels
    if not {"user_id", "sparse"}.issubset(labels.columns):
        raise ValueError("sparse labels must contain user_id and sparse")
    eligible = set(ledger.table.loc[ledger.table["raw_evaluation_eligible"], "user_id"].astype(int))
    supplied = labels.loc[labels["user_id"].isin(eligible), "user_id"].astype(int)
    missing = sorted(eligible - set(supplied))
    if missing:
        raise ValueError(f"missing frozen sparse labels for eligible users: {missing[:10]}")
    if supplied.duplicated().any() or labels.loc[labels["user_id"].isin(eligible), "sparse"].isna().any():
        raise ValueError("frozen sparse labels must be complete and non-null for eligible users")


@dataclass(frozen=True)
class GlobalPopularity:
    scores: Mapping[int, float]
    total_activity: float

    def score_candidates(self, candidate_artists: Iterable[int]) -> pd.DataFrame:
        return _ranked(self.scores, candidate_artists)


def fit_global_popularity(fitting_interactions: pd.DataFrame) -> GlobalPopularity:
    """Fit global popularity from fitting rows only using ``log1p(weight)``."""
    table = _fit_table(fitting_interactions)
    scores = table.groupby("artist_id", sort=True)["activity"].sum().astype(float).to_dict()
    return GlobalPopularity({int(k): float(v) for k, v in scores.items()}, float(table["activity"].sum()))


@dataclass(frozen=True)
class CommunityPopularity:
    scores_by_community: Mapping[int, Mapping[int, float]]
    global_scores: Mapping[int, float]
    artist_universe: tuple[int, ...]
    beta: float

    def scores_for_user(self, user_id: int, membership: pd.DataFrame) -> Mapping[int, float]:
        labels = membership.set_index("user_id")["community_id"]
        if int(user_id) not in labels.index:
            raise KeyError(f"user {user_id} has no community assignment")
        return self.scores_by_community.get(int(labels.loc[int(user_id)]), self.global_scores)

    def score_candidates(
        self, user_id: int, candidate_artists: Iterable[int], membership: pd.DataFrame
    ) -> pd.DataFrame:
        return _ranked(self.scores_for_user(user_id, membership), candidate_artists)


def fit_community_popularity(
    fitting_interactions: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    beta: float,
) -> CommunityPopularity:
    """Fit community popularity with frozen shrinkage/backoff from fit rows."""
    if beta not in BETA_VALUES:
        raise ValueError(f"beta must be one of {BETA_VALUES}")
    if tuple(membership.columns) != MEMBERSHIP_COLUMNS:
        raise ValueError("membership must have columns user_id, community_id")
    table = _fit_table(fitting_interactions)
    if membership["user_id"].duplicated().any():
        raise ValueError("membership contains duplicate users")
    if membership[["user_id", "community_id"]].isna().any().any():
        raise ValueError("membership assignments must be non-null")
    fitting_users = set(table["user_id"].astype(int))
    membership_users = set(membership["user_id"].astype(int))
    missing_users = sorted(fitting_users - membership_users)
    if missing_users:
        raise ValueError(
            "missing community assignments for fitting users: "
            f"{missing_users[:10]}"
        )
    universe = tuple(sorted(table["artist_id"].astype(int).unique()))
    activity_by_artist = table.groupby("artist_id", sort=True)["activity"].sum()
    total = float(activity_by_artist.sum())
    if total > 0:
        global_scores = {artist: float(activity_by_artist.get(artist, 0.0) / total) for artist in universe}
    else:
        uniform = 1.0 / len(universe) if universe else 0.0
        global_scores = {artist: uniform for artist in universe}
    with_community = table.merge(membership, on="user_id", how="left", validate="many_to_one")
    community_totals = with_community.groupby("community_id", dropna=True)["activity"].sum()
    community_artist = with_community.groupby(["community_id", "artist_id"], sort=True)["activity"].sum()
    result: dict[int, dict[int, float]] = {}
    for community_id in sorted(membership["community_id"].astype(int).unique()):
        mass = float(community_totals.get(community_id, 0.0))
        result[community_id] = {
            artist: float((community_artist.get((community_id, artist), 0.0) + beta * global_scores[artist]) / (mass + beta))
            for artist in universe
        }
    return CommunityPopularity(result, global_scores, universe, float(beta))


__all__ = [
    "BETA_VALUES", "GlobalPopularity", "CommunityPopularity", "fit_global_popularity",
    "fit_community_popularity", "validate_complete_sparse_labels",
]
