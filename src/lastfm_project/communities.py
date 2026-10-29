"""Deterministic friendship communities for fixed ``lastfm-v1``."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import networkx as nx
import pandas as pd

from .data_loading import FriendshipTables

COMMUNITY_SEED = 5101
MEMBERSHIP_COLUMNS = ("user_id", "community_id")


def _edges(friendships: FriendshipTables | pd.DataFrame) -> pd.DataFrame:
    edges = friendships.undirected if isinstance(friendships, FriendshipTables) else friendships
    if tuple(edges.columns) != ("user_low_id", "user_high_id"):
        raise ValueError("edges must have columns user_low_id, user_high_id")
    result = edges[["user_low_id", "user_high_id"]].astype("int64").copy()
    result["user_low_id"], result["user_high_id"] = (
        result[["user_low_id", "user_high_id"]].min(axis=1),
        result[["user_low_id", "user_high_id"]].max(axis=1),
    )
    return result.drop_duplicates().sort_values(
        ["user_low_id", "user_high_id"], kind="mergesort", ignore_index=True
    )


def build_friendship_graph(
    user_ids: Iterable[int], friendships: FriendshipTables | pd.DataFrame
) -> nx.Graph:
    """Build a simple graph with every audited user, including isolates."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted({int(user_id) for user_id in user_ids}))
    graph.add_edges_from(_edges(friendships).itertuples(index=False, name=None))
    return graph


@dataclass(frozen=True)
class CommunityResult:
    graph: nx.Graph
    membership: pd.DataFrame
    modularity: float
    community_count: int
    size_distribution: tuple[int, ...]

    @property
    def membership_counts(self) -> dict[int, int]:
        return self.membership["community_id"].value_counts(sort=False).sort_index().astype(int).to_dict()


def detect_communities(
    user_ids: Iterable[int], friendships: FriendshipTables | pd.DataFrame, *, seed: int = COMMUNITY_SEED
) -> CommunityResult:
    """Run NetworkX Louvain and canonicalize each label to its minimum user ID."""
    graph = build_friendship_graph(user_ids, friendships)
    raw = nx.community.louvain_communities(graph, seed=seed)
    communities = sorted((tuple(sorted(int(user) for user in group)) for group in raw), key=lambda group: group[0])
    rows = [(user_id, group[0]) for group in communities for user_id in group]
    membership = pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).sort_values("user_id", kind="mergesort", ignore_index=True)
    membership = membership.astype("int64")
    modularity = nx.community.modularity(graph, [set(group) for group in communities])
    sizes = tuple(sorted((len(group) for group in communities), reverse=True))
    return CommunityResult(graph, membership, float(modularity), len(communities), sizes)


def shuffle_community_assignments(
    membership: pd.DataFrame, *, seed: int = COMMUNITY_SEED
) -> pd.DataFrame:
    """Permute labels across sorted users while preserving label-size counts."""
    if tuple(membership.columns) != MEMBERSHIP_COLUMNS:
        raise ValueError("membership must have columns user_id, community_id")
    source = membership.sort_values("user_id", kind="mergesort").reset_index(drop=True)
    if source["user_id"].duplicated().any():
        raise ValueError("membership contains duplicate users")
    user_ids = source["user_id"].astype(int).tolist()
    permuted = user_ids.copy()
    random.Random(seed).shuffle(permuted)
    labels = source.set_index("user_id")["community_id"]
    result = pd.DataFrame({"user_id": user_ids, "community_id": [int(labels[user]) for user in permuted]})
    return result.astype("int64")


__all__ = [
    "COMMUNITY_SEED", "MEMBERSHIP_COLUMNS", "CommunityResult", "build_friendship_graph",
    "detect_communities", "shuffle_community_assignments",
]
