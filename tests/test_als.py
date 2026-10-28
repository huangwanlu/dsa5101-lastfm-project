from __future__ import annotations

from math import log1p

import numpy as np
import pandas as pd
import pytest

from lastfm_project.als import ALSConfig, ALSModel, build_als_input, fit_implicit_als


def interactions(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "artist_id", "weight"]).astype("int64")


def test_sparse_input_is_user_by_artist_with_sorted_raw_id_mappings_and_exact_confidence() -> None:
    source = interactions([(20, 300, 3), (10, 200, 1), (20, 100, 7)])
    config = ALSConfig(32, 0.01, 10.0)
    prepared = build_als_input(source.sample(frac=1, random_state=4), config)
    assert prepared.user_ids == (10, 20)
    assert prepared.artist_ids == (100, 200, 300)
    assert prepared.confidence.shape == (2, 3)
    assert prepared.confidence.nnz == 3
    assert prepared.confidence[0, 1] == pytest.approx(1 + 10 * log1p(1), rel=1e-6)
    assert prepared.confidence[1, 0] == pytest.approx(1 + 10 * log1p(7), rel=1e-6)


def test_library_alpha_is_one_so_protocol_alpha_is_not_applied_twice() -> None:
    captured: dict[str, object] = {}

    class FakeALS:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.user_factors = None
            self.item_factors = None

        def fit(self, matrix, show_progress=False):
            captured["matrix"] = matrix.copy()
            self.user_factors = np.ones((matrix.shape[0], 32), dtype=np.float32)
            self.item_factors = np.ones((matrix.shape[1], 32), dtype=np.float32)

    source = interactions([(2, 20, 4), (1, 10, 2)])
    fit_implicit_als(source, ALSConfig(32, 0.01, 10.0), model_factory=FakeALS)
    assert captured["alpha"] == 1.0
    assert captured["num_threads"] == 1
    matrix = captured["matrix"]
    assert matrix[0, 0] == pytest.approx(1 + 10 * log1p(2), rel=1e-6)


def test_mf_scores_use_raw_id_mappings_and_missing_items_stay_missing() -> None:
    model = ALSModel(
        ALSConfig(32, 0.01, 1.0),
        (10,),
        (100, 300),
        {10: 0},
        {100: 0, 300: 1},
        np.pad(np.array([[2.0, 3.0]]), ((0, 0), (0, 30))),
        np.pad(np.array([[5.0, 7.0], [11.0, 13.0]]), ((0, 0), (0, 30))),
    )
    scores, represented = model.score(10, [300, 200, 100])
    assert scores[[0, 2]].tolist() == [61.0, 31.0]
    assert np.isneginf(scores[1])
    assert represented.tolist() == [True, False, True]


def test_nonfinite_learned_factors_fail_closed() -> None:
    class BadALS:
        def __init__(self, **kwargs):
            self.user_factors = self.item_factors = None

        def fit(self, matrix, show_progress=False):
            self.user_factors = np.full((matrix.shape[0], 32), np.nan, dtype=np.float32)
            self.item_factors = np.ones((matrix.shape[1], 32), dtype=np.float32)

    with pytest.raises(FloatingPointError, match="nonfinite learned factors"):
        fit_implicit_als(interactions([(1, 10, 1)]), ALSConfig(32, 0.01, 1.0), model_factory=BadALS)

