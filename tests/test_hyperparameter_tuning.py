"""Unit tests for hyperparameter tuning utilities."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from backend.ml.bgnbd_model import BGNBDModel
from backend.ml.hyperparameter_tuning import tune_penalizer_grid


def _make_rfm(n: int = 150, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t_days = rng.integers(180, 730, size=n)
    frequency = rng.integers(0, 10, size=n)
    recency_days = np.array([
        float(rng.integers(0, max(1, t - 1))) if t > 1 else 0.0
        for t in t_days
    ])
    monetary_avg = rng.exponential(scale=40, size=n).clip(1, 2000)
    return pl.DataFrame({
        "customer_id":    [f"C{i:05d}" for i in range(n)],
        "frequency":      frequency.astype(int).tolist(),
        "recency_days":   recency_days.tolist(),
        "t_days":         t_days.astype(int).tolist(),
        "monetary_avg":   monetary_avg.tolist(),
        "monetary_total": (monetary_avg * (frequency + 1)).tolist(),
    })


def test_grid_search_returns_best_penalizer() -> None:
    cal = _make_rfm(150, seed=1)
    hold = _make_rfm(100, seed=2)
    best, results = tune_penalizer_grid(
        cal, hold,
        observation_end=date(2011, 6, 30),
        penalizer_values=[0.001, 0.01, 0.1],
    )
    assert isinstance(best, float)
    assert best > 0
    assert best in [0.001, 0.01, 0.1]


def test_grid_search_results_dataframe() -> None:
    cal = _make_rfm(150, seed=3)
    hold = _make_rfm(100, seed=4)
    _, results = tune_penalizer_grid(
        cal, hold,
        observation_end=date(2011, 6, 30),
        penalizer_values=[0.001, 0.1],
    )
    assert "penalizer" in results.columns
    assert "mae_pct_12m" in results.columns
    assert "gini_coefficient" in results.columns
    assert len(results) == 2


def test_grid_search_results_sorted_by_mae() -> None:
    cal = _make_rfm(150, seed=5)
    hold = _make_rfm(100, seed=6)
    _, results = tune_penalizer_grid(
        cal, hold,
        observation_end=date(2011, 6, 30),
        penalizer_values=[0.001, 0.01, 0.1],
    )
    maes = results["mae_pct_12m"].to_list()
    assert maes == sorted(maes)