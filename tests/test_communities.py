from __future__ import annotations

import pandas as pd

from lastfm_project.communities import build_friendship_graph, detect_communities, shuffle_community_assignments


def edges(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_low_id", "user_high_id"]).astype("int64")


def test_graph_canonicalizes_duplicate_undirected_edges_and_keeps_isolates() -> None:
    graph = build_friendship_graph([1, 2, 3, 4], edges([(2, 1), (1, 2), (2, 3)]))
    assert sorted(graph.nodes) == [1, 2, 3, 4]
    assert sorted(tuple(sorted(edge)) for edge in graph.edges) == [(1, 2), (2, 3)]


def test_louvain_is_deterministic_and_community_ids_are_minimum_user_ids() -> None:
    result = detect_communities(range(1, 7), edges([(1, 2), (2, 3), (4, 5)]))
    again = detect_communities(range(1, 7), edges([(4, 5), (2, 3), (1, 2)]))
    pd.testing.assert_frame_equal(result.membership, again.membership)
    assert set(result.membership["community_id"]) == {1, 4, 6}
    assert result.membership_counts == {1: 3, 4: 2, 6: 1}
    assert result.size_distribution == (3, 2, 1)
    assert result.community_count == 3


def test_shuffled_assignments_preserve_community_sizes_and_are_seeded() -> None:
    membership = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "community_id": [1, 1, 1, 4, 4]}).astype("int64")
    shuffled = shuffle_community_assignments(membership)
    assert shuffled.groupby("community_id").size().to_dict() == {1: 3, 4: 2}
    pd.testing.assert_frame_equal(shuffled, shuffle_community_assignments(membership.sample(frac=1, random_state=8)))
