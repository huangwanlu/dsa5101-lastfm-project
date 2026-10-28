"""Implicit ALS for the frozen ``lastfm-v1`` protocol.

``implicit`` 0.7.2 expects a user-by-item CSR whose stored values are the
*complete* positive-entry confidences used by its solver.  We therefore store
``1 + alpha * log1p(weight)`` and leave the library's own ``alpha`` at one.
This implements the frozen confidence exactly and avoids double scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse


ALS_FACTORS = (32, 64)
ALS_REGULARIZATIONS = (0.01, 0.10)
ALS_ALPHAS = (1.0, 10.0)
ALS_ITERATIONS = 20
ALS_SEED = 5101


@dataclass(frozen=True, order=True)
class ALSConfig:
    factors: int
    regularization: float
    alpha: float
    iterations: int = ALS_ITERATIONS
    seed: int = ALS_SEED

    def __post_init__(self) -> None:
        if self.factors not in ALS_FACTORS:
            raise ValueError(f"factors must be one of {ALS_FACTORS}")
        if self.regularization not in ALS_REGULARIZATIONS:
            raise ValueError(f"regularization must be one of {ALS_REGULARIZATIONS}")
        if self.alpha not in ALS_ALPHAS:
            raise ValueError(f"alpha must be one of {ALS_ALPHAS}")
        if self.iterations != ALS_ITERATIONS or self.seed != ALS_SEED:
            raise ValueError("iterations and seed are frozen at 20 and 5101")


ALS_GRID = tuple(
    ALSConfig(factors, regularization, alpha)
    for factors in ALS_FACTORS
    for regularization in ALS_REGULARIZATIONS
    for alpha in ALS_ALPHAS
)


@dataclass(frozen=True)
class ALSInput:
    confidence: sparse.csr_matrix
    user_ids: tuple[int, ...]
    artist_ids: tuple[int, ...]
    user_to_index: dict[int, int]
    artist_to_index: dict[int, int]


@dataclass(frozen=True)
class ALSModel:
    config: ALSConfig
    user_ids: tuple[int, ...]
    artist_ids: tuple[int, ...]
    user_to_index: dict[int, int]
    artist_to_index: dict[int, int]
    user_factors: np.ndarray
    item_factors: np.ndarray
    library_alpha: float = 1.0

    def score(self, user_id: int, artist_ids: tuple[int, ...] | list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return scores and a mask indicating a learned representation.

        A missing user or item is expected absence and receives ``-inf`` plus
        ``False``. Existing factors that yield nonfinite values raise instead.
        """
        requested = np.asarray(artist_ids, dtype=np.int64)
        scores = np.full(len(requested), -np.inf, dtype=np.float64)
        represented = np.zeros(len(requested), dtype=bool)
        user_index = self.user_to_index.get(int(user_id))
        if user_index is None:
            return scores, represented
        item_indexes = np.fromiter(
            (self.artist_to_index.get(int(artist), -1) for artist in requested),
            dtype=np.int64,
            count=len(requested),
        )
        represented = item_indexes >= 0
        if represented.any():
            values = self.item_factors[item_indexes[represented]] @ self.user_factors[user_index]
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite ALS score for user {user_id}")
            scores[represented] = values.astype(np.float64, copy=False)
        return scores, represented


def build_als_input(fitting_interactions: pd.DataFrame, config: ALSConfig) -> ALSInput:
    """Build deterministic mappings and a sparse user-by-artist confidence matrix."""
    required = {"user_id", "artist_id", "weight"}
    if not required.issubset(fitting_interactions.columns):
        raise ValueError("fitting interactions must contain user_id, artist_id, weight")
    table = fitting_interactions[["user_id", "artist_id", "weight"]].copy()
    if table.empty:
        raise ValueError("cannot fit ALS without interactions")
    if table.duplicated(["user_id", "artist_id"]).any():
        raise ValueError("fitting interactions contain duplicate pairs")
    weights = table["weight"].to_numpy(dtype=np.float64)
    if (weights <= 0).any() or not np.isfinite(weights).all():
        raise ValueError("fitting interaction weights must be positive and finite")
    user_ids = tuple(sorted(int(value) for value in table["user_id"].unique()))
    artist_ids = tuple(sorted(int(value) for value in table["artist_id"].unique()))
    user_to_index = {raw_id: index for index, raw_id in enumerate(user_ids)}
    artist_to_index = {raw_id: index for index, raw_id in enumerate(artist_ids)}
    rows = table["user_id"].map(user_to_index).to_numpy(dtype=np.int32)
    columns = table["artist_id"].map(artist_to_index).to_numpy(dtype=np.int32)
    confidence = 1.0 + config.alpha * np.log1p(weights)
    if not np.isfinite(confidence).all():
        raise FloatingPointError("nonfinite ALS confidence")
    matrix = sparse.coo_matrix(
        (confidence.astype(np.float32), (rows, columns)),
        shape=(len(user_ids), len(artist_ids)),
        dtype=np.float32,
    ).tocsr()
    matrix.sort_indices()
    return ALSInput(matrix, user_ids, artist_ids, user_to_index, artist_to_index)


def fit_implicit_als(
    fitting_interactions: pd.DataFrame,
    config: ALSConfig,
    *,
    show_progress: bool = False,
    model_factory: Callable[..., object] | None = None,
) -> ALSModel:
    """Fit CPU implicit ALS with deterministic initialization and one thread."""
    if model_factory is None:
        from implicit.cpu.als import AlternatingLeastSquares

        model_factory = AlternatingLeastSquares
    prepared = build_als_input(fitting_interactions, config)
    implementation = model_factory(
        factors=config.factors,
        regularization=config.regularization,
        # Stored CSR values already equal 1 + protocol alpha * activity.
        alpha=1.0,
        iterations=config.iterations,
        random_state=config.seed,
        dtype=np.float32,
        num_threads=1,
    )
    implementation.fit(prepared.confidence, show_progress=show_progress)
    user_factors = np.asarray(implementation.user_factors)
    item_factors = np.asarray(implementation.item_factors)
    expected_user_shape = (len(prepared.user_ids), config.factors)
    expected_item_shape = (len(prepared.artist_ids), config.factors)
    if user_factors.shape != expected_user_shape or item_factors.shape != expected_item_shape:
        raise RuntimeError(
            f"implicit returned factor shapes {user_factors.shape}, {item_factors.shape}; "
            f"expected {expected_user_shape}, {expected_item_shape}"
        )
    if not np.isfinite(user_factors).all() or not np.isfinite(item_factors).all():
        raise FloatingPointError("implicit ALS returned nonfinite learned factors")
    return ALSModel(
        config=config,
        user_ids=prepared.user_ids,
        artist_ids=prepared.artist_ids,
        user_to_index=prepared.user_to_index,
        artist_to_index=prepared.artist_to_index,
        user_factors=user_factors.copy(),
        item_factors=item_factors.copy(),
    )


__all__ = [
    "ALS_ALPHAS", "ALS_FACTORS", "ALS_GRID", "ALS_ITERATIONS", "ALS_REGULARIZATIONS",
    "ALS_SEED", "ALSConfig", "ALSInput", "ALSModel", "build_als_input", "fit_implicit_als",
]
