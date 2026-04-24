"""
Unit tests for BGNBDModel.

Tests:
  - Model fitting on synthetic data
  - Prediction output shapes and value ranges
  - Single-customer scoring
  - Param persistence and loading
  - Confidence interval validity
  - Metric computation functions
"""

from __future__ import annotations

import pickle
import tempfile
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from backend.ml.bgnbd_model import (
    BGNBDModel,
    compute_gini,
    compute_top_decile_lift,
    compute_calibration_error,
    polars_rfm_to_pandas,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_rfm(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """Generate a realistic synthetic RFM DataFrame for testing."""
    rng = np.random.default_rng(seed)
    t_days = rng.integers(180, 730, size=n)
    frequency = rng.integers(0, 15, size=n)
    recency_days = np.array([
        float(rng.integers(0, max(1, t - 1))) if t > 1 else 0.0
        for t in t_days
    ])
    monetary_avg = rng.exponential(scale=50, size=n).clip(1, 5000)

    return pl.DataFrame({
        "customer_id":    [f"C{i:05d}" for i in range(n)],
        "frequency":      frequency.astype(int).tolist(),
        "recency_days":   recency_days.tolist(),
        "t_days":         t_days.astype(int).tolist(),
        "monetary_avg":   monetary_avg.tolist(),
        "monetary_total": (monetary_avg * (frequency + 1)).tolist(),
    })


@pytest.fixture
def rfm_df() -> pl.DataFrame:
    return _make_rfm(300)


@pytest.fixture
def fitted_model(rfm_df: pl.DataFrame) -> BGNBDModel:
    model = BGNBDModel(
        penalizer_coef=0.01,
        model_version=f"test_{uuid.uuid4().hex[:6]}",
        observation_end=date(2011, 6, 30),
    )
    model.fit(rfm_df)
    return model


# ─────────────────────────────────────────────────────────────
# Conversion tests
# ─────────────────────────────────────────────────────────────

def test_polars_to_pandas_columns(rfm_df: pl.DataFrame) -> None:
    pd_df = polars_rfm_to_pandas(rfm_df)
    assert "frequency" in pd_df.columns
    assert "recency" in pd_df.columns
    assert "T" in pd_df.columns
    assert "monetary_value" in pd_df.columns
    assert pd_df.index.name == "customer_id"


def test_polars_to_pandas_recency_leq_T(rfm_df: pl.DataFrame) -> None:
    pd_df = polars_rfm_to_pandas(rfm_df)
    assert (pd_df["recency"] <= pd_df["T"]).all()


def test_polars_to_pandas_no_negative_values(rfm_df: pl.DataFrame) -> None:
    pd_df = polars_rfm_to_pandas(rfm_df)
    assert (pd_df["frequency"] >= 0).all()
    assert (pd_df["monetary_value"] > 0).all()


# ─────────────────────────────────────────────────────────────
# Fitting tests
# ─────────────────────────────────────────────────────────────

def test_fit_sets_bgf_and_ggf(fitted_model: BGNBDModel) -> None:
    assert fitted_model.bgf is not None
    assert fitted_model.ggf is not None
    assert fitted_model._is_fitted


def test_fit_stores_params(fitted_model: BGNBDModel) -> None:
    params = fitted_model.get_params()
    assert "r" in params["bgnbd"]
    assert "alpha" in params["bgnbd"]
    assert "a" in params["bgnbd"]
    assert "b" in params["bgnbd"]
    assert "p" in params["gamma_gamma"]
    assert "q" in params["gamma_gamma"]
    assert "v" in params["gamma_gamma"]


def test_fit_params_positive(fitted_model: BGNBDModel) -> None:
    params = fitted_model.get_params()
    for v in params["bgnbd"].values():
        assert v > 0
    for v in params["gamma_gamma"].values():
        assert v > 0


def test_unfit_model_raises(rfm_df: pl.DataFrame) -> None:
    model = BGNBDModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(rfm_df)


# ─────────────────────────────────────────────────────────────
# Prediction tests
# ─────────────────────────────────────────────────────────────

def test_predict_returns_polars(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    preds = fitted_model.predict(rfm_df, n_bootstrap=10)
    assert isinstance(preds, pl.DataFrame)


def test_predict_one_row_per_customer(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    preds = fitted_model.predict(rfm_df, n_bootstrap=10)
    assert len(preds) == len(polars_rfm_to_pandas(rfm_df))


def test_predict_ltv_non_negative(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    preds = fitted_model.predict(rfm_df, n_bootstrap=10)
    assert (preds["ltv_12m"] >= 0).all()
    assert (preds["ltv_24m"] >= 0).all()
    assert (preds["ltv_36m"] >= 0).all()


def test_predict_probability_alive_in_range(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    preds = fitted_model.predict(rfm_df, n_bootstrap=10)
    assert (preds["probability_alive"] >= 0).all()
    assert (preds["probability_alive"] <= 1).all()


def test_predict_ltv_ordering(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    """LTV should increase with horizon: 12m <= 24m <= 36m."""
    preds = fitted_model.predict(rfm_df, n_bootstrap=10)
    assert (preds["ltv_36m"] >= preds["ltv_12m"]).all()
    assert (preds["ltv_24m"] >= preds["ltv_12m"]).all()


def test_predict_ci_lower_leq_upper(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    preds = fitted_model.predict(rfm_df, n_bootstrap=20)
    assert (preds["ltv_36m_upper"] >= preds["ltv_36m_lower"]).all()


# ─────────────────────────────────────────────────────────────
# Single customer scoring
# ─────────────────────────────────────────────────────────────

def test_predict_single_returns_dict(fitted_model: BGNBDModel) -> None:
    result = fitted_model.predict_single(
        frequency=5, recency_days=90, t_days=180, monetary_avg=45.0
    )
    assert isinstance(result, dict)
    assert "ltv_12m" in result
    assert "ltv_36m" in result
    assert "probability_alive" in result


def test_predict_single_non_negative(fitted_model: BGNBDModel) -> None:
    result = fitted_model.predict_single(
        frequency=2, recency_days=60, t_days=200, monetary_avg=30.0
    )
    for v in result.values():
        assert v >= 0


def test_predict_single_p_alive_in_range(fitted_model: BGNBDModel) -> None:
    result = fitted_model.predict_single(
        frequency=10, recency_days=30, t_days=365, monetary_avg=50.0
    )
    assert 0 <= result["probability_alive"] <= 1


# ─────────────────────────────────────────────────────────────
# Serialisation tests
# ─────────────────────────────────────────────────────────────

def test_save_and_load(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fitted_model.save_to_disk(tmpdir)
        loaded = BGNBDModel.load_from_disk(tmpdir, fitted_model.model_version)

    assert loaded._is_fitted
    preds_orig   = fitted_model.predict(rfm_df, n_bootstrap=5)
    preds_loaded = loaded.predict(rfm_df, n_bootstrap=5)

    # LTV predictions should be identical after reload
    diff = (preds_orig["ltv_36m"] - preds_loaded["ltv_36m"]).abs().max()
    assert diff < 0.01, f"Max LTV diff after reload: {diff}"


# ─────────────────────────────────────────────────────────────
# Metric function tests
# ─────────────────────────────────────────────────────────────

def test_gini_perfect_ranking() -> None:
    y_true = np.array([10, 20, 30, 40, 50], dtype=float)
    y_pred = np.array([10, 20, 30, 40, 50], dtype=float)  # perfect
    g = compute_gini(y_true, y_pred)
    assert g > 0.5  # should be high for perfect ranking


def test_gini_random_ranking() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 100, 1000)
    y_pred = rng.uniform(0, 100, 1000)  # random
    g = compute_gini(y_true, y_pred)
    assert -0.2 < g < 0.2  # near-zero for random


def test_top_decile_lift_perfect() -> None:
    n = 100
    y_true = np.zeros(n)
    y_pred = np.zeros(n)
    # Top 10 are best
    y_true[:10] = 100.0
    y_pred[:10] = 100.0
    lift = compute_top_decile_lift(y_true, y_pred)
    assert lift > 1.0


def test_calibration_error_perfect() -> None:
    y_true = np.array([1, 2, 3, 4, 5] * 20, dtype=float)
    y_pred = y_true.copy()  # perfect predictions
    err = compute_calibration_error(y_true, y_pred)
    assert err < 0.01


def test_calibration_error_bad() -> None:
    y_true = np.ones(100) * 100
    y_pred = np.ones(100) * 1   # 100× off
    err = compute_calibration_error(y_true, y_pred)
    assert err > 0.5


# ─────────────────────────────────────────────────────────────
# Calibration plot data
# ─────────────────────────────────────────────────────────────

def test_calibration_plot_data_shape(fitted_model: BGNBDModel, rfm_df: pl.DataFrame) -> None:
    holdout = _make_rfm(200, seed=99)  # simulated holdout
    cal_data = fitted_model.get_calibration_plot_data(rfm_df, holdout, n_buckets=5)
    assert "predicted_purchases_avg" in cal_data.columns
    assert "actual_purchases_avg" in cal_data.columns
    assert len(cal_data) <= 5


# ─────────────────────────────────────────────────────────────
# Probability-alive matrix
# ─────────────────────────────────────────────────────────────

def test_p_alive_matrix_values_in_range(fitted_model: BGNBDModel) -> None:
    matrix = fitted_model.get_probability_alive_matrix(
        max_frequency=20, max_recency_days=100, t_days=200, step=10
    )
    assert (matrix["p_alive"] >= 0).all()
    assert (matrix["p_alive"] <= 1).all()


def test_p_alive_matrix_has_required_columns(fitted_model: BGNBDModel) -> None:
    matrix = fitted_model.get_probability_alive_matrix(step=20, max_frequency=20)
    assert "frequency" in matrix.columns
    assert "recency_days" in matrix.columns
    assert "p_alive" in matrix.columns