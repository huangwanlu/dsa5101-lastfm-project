from __future__ import annotations

import numpy as np
import pandas as pd

from lastfm_project.als import ALS_GRID, ALSConfig, ALSModel
from lastfm_project.evaluation_ledger import build_phase_ledger
from lastfm_project.recommendation import MetricEvaluation
from lastfm_project.validation import (
    HYBRID_LAMBDAS,
    ValidationRecord,
    run_validation_selection,
    select_best,
)


def record(stage: str, parameters: dict[str, float | int], ndcg: float = 0.5) -> ValidationRecord:
    return ValidationRecord(stage, parameters, ndcg, 0.25, 1, (), 0.01)


def test_fixed_grids_and_deterministic_tie_breaks() -> None:
    assert len(ALS_GRID) == 8
    assert HYBRID_LAMBDAS == (0.25, 0.5, 0.75)
    als = [record("als", {"factors": f, "regularization": r, "alpha": a}) for f in (32, 64) for r in (0.01, 0.1) for a in (1, 10)]
    assert select_best(als, stage="als").parameters == {"factors": 32, "regularization": 0.1, "alpha": 1}
    hybrid = [record("hybrid", {"beta": b, "lambda": l}) for b in (10, 100, 1000) for l in HYBRID_LAMBDAS]
    assert select_best(hybrid, stage="hybrid").parameters == {"beta": 10, "lambda": 0.25}


def test_staged_selection_runs_exact_8_3_9_grid_without_test_data(monkeypatch) -> None:
    rows = [
        (1, 1, 1, "train"), (1, 2, 1, "validation"), (1, 3, 999, "test"),
        (2, 2, 1, "train"), (2, 1, 1, "validation"), (2, 4, 999, "test"),
    ]
    split = pd.DataFrame(rows, columns=["user_id", "artist_id", "weight", "split"]).astype(
        {"user_id": "int64", "artist_id": "int64", "weight": "int64", "split": "string"}
    )
    sparse = pd.DataFrame({"user_id": [1, 2], "sparse": pd.array([True, True], dtype="boolean")})
    ledger = build_phase_ledger(split, sparse, phase="validation")
    membership = pd.DataFrame({"user_id": [1, 2], "community_id": [1, 1]}).astype("int64")
    seen_fit_weights: list[list[int]] = []

    def fake_fit(fitting, config, show_progress=False):
        seen_fit_weights.append(fitting["weight"].astype(int).tolist())
        return ALSModel(config, (1, 2), (1, 2), {1: 0, 2: 1}, {1: 0, 2: 1}, np.ones((2, config.factors)), np.ones((2, config.factors)))

    metric = MetricEvaluation(0.5, 0.25, pd.DataFrame(), (1, 2), (), {})
    monkeypatch.setattr("lastfm_project.validation.fit_implicit_als", fake_fit)
    monkeypatch.setattr("lastfm_project.validation._evaluate_als", lambda ledger, model: metric)
    monkeypatch.setattr("lastfm_project.validation.fit_community_popularity", lambda fitting, membership, beta: object())
    monkeypatch.setattr("lastfm_project.validation._evaluate_community", lambda ledger, profile, membership: metric)
    monkeypatch.setattr(
        "lastfm_project.validation._evaluate_hybrid_lambdas",
        lambda ledger, model, profile, membership: {value: metric for value in HYBRID_LAMBDAS},
    )

    result = run_validation_selection(ledger, membership)
    assert [record.stage for record in result.records].count("als") == 8
    assert [record.stage for record in result.records].count("community") == 3
    assert [record.stage for record in result.records].count("hybrid") == 9
    assert all(999 not in weights for weights in seen_fit_weights)
    assert result.best_als.parameters["factors"] == 32
    assert result.best_als.parameters["regularization"] == 0.1
    assert result.best_hybrid.parameters == {"beta": 10.0, "lambda": 0.25}
    assert result.h3_estimable is False


def test_selection_rejects_a_test_phase_ledger() -> None:
    split = pd.DataFrame(
        [(1, 1, 1, "train"), (1, 2, 1, "test")],
        columns=["user_id", "artist_id", "weight", "split"],
    ).astype({"user_id": "int64", "artist_id": "int64", "weight": "int64", "split": "string"})
    ledger = build_phase_ledger(split, phase="test")
    membership = pd.DataFrame({"user_id": [1], "community_id": [1]}).astype("int64")
    try:
        run_validation_selection(ledger, membership)
    except ValueError as exc:
        assert "validation-phase" in str(exc)
    else:
        raise AssertionError("test-phase tuning was not rejected")

