"""Unit tests for CausalLTVPipeline and DoubleMLEstimator."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from backend.ml.causal_model import (
    DoubleMLEstimator,
    prepare_causal_dataset,
    CONTROL_FEATURES,
    TREATMENT_DEFINITIONS,
)


def _make_rfm(n: int = 300, seed: int = 42) -> pl.DataFrame:
    """Synthetic RFM with known LTV-driving features."""
    rng = np.random.default_rng(seed)
    t_days    = rng.integers(180, 730, n)
    frequency = rng.integers(0, 10, n)
    monetary  = rng.exponential(50, n).clip(5, 2000)

    # days_to_second_purchase: smaller → "onboarding done" treatment
    d2s = rng.choice([5, 7, 14, 30, 60, 120, 365], n)

    # Actual LTV: partially caused by fast second purchase
    base_ltv  = monetary * (frequency + 1)
    treatment = (d2s <= 7).astype(float)
    ltv       = (base_ltv + treatment * rng.uniform(50, 200, n) + rng.normal(0, 50, n)).clip(0)

    return pl.DataFrame({
        "customer_id":              [f"C{i:05d}" for i in range(n)],
        "frequency":                frequency.astype(int).tolist(),
        "recency_days":             rng.uniform(0, t_days).tolist(),
        "t_days":                   t_days.astype(int).tolist(),
        "monetary_avg":             monetary.tolist(),
        "monetary_total":           (monetary * (frequency + 1)).tolist(),
        "monetary_std":             rng.uniform(0, 20, n).tolist(),
        "orders_count":             frequency.astype(int).tolist(),
        "unique_products":          rng.integers(1, 10, n).astype(int).tolist(),
        "unique_categories":        rng.integers(1, 5, n).astype(int).tolist(),
        "avg_days_between_orders":  rng.uniform(10, 100, n).tolist(),
        "days_to_second_purchase":  d2s.tolist(),
        "first_purchase_amount":    monetary.tolist(),
        "multi_country":            rng.choice([True, False], n).tolist(),
        "actual_ltv_12m":           ltv.tolist(),
    })


def test_prepare_causal_dataset_returns_dataframe() -> None:
    rfm = _make_rfm(200)
    df, treatments, controls = prepare_causal_dataset(rfm)
    assert len(df) > 0
    assert len(treatments) > 0
    assert len(controls) > 0


def test_prepare_causal_dataset_no_nulls() -> None:
    rfm = _make_rfm(200)
    df, treatments, controls = prepare_causal_dataset(rfm)
    for col in controls:
        assert df[col].isna().sum() == 0, f"NaN in control {col}"


def test_prepare_causal_dataset_treatment_binary() -> None:
    rfm = _make_rfm(200)
    df, treatments, _ = prepare_causal_dataset(rfm)
    for t in treatments:
        col = f"t_{t}"
        assert set(df[col].unique()).issubset({0.0, 1.0}), f"{col} not binary"


def test_prepare_causal_dataset_log_ltv() -> None:
    rfm = _make_rfm(200)
    df, _, _ = prepare_causal_dataset(rfm)
    assert "log_ltv" in df.columns
    assert (df["log_ltv"] >= 0).all()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("econml"),
    reason="econml not installed",
)
def test_double_ml_fits_and_predicts() -> None:
    rfm = _make_rfm(300)
    df, _, controls = prepare_causal_dataset(rfm)

    est = DoubleMLEstimator(
        treatment_name="onboarding_completed",
        treatment_type="binary",
        cv_folds=3,
    )
    est.fit(df, controls, outcome_col="log_ltv")

    assert est._is_fitted
    cate = est.estimate_cate(df, controls)
    assert len(cate) == len(df)
    assert isinstance(cate, np.ndarray)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("econml"),
    reason="econml not installed",
)
def test_double_ml_ate_is_finite() -> None:
    rfm = _make_rfm(300)
    df, _, controls = prepare_causal_dataset(rfm)
    est = DoubleMLEstimator("onboarding_completed", cv_folds=3)
    est.fit(df, controls)
    assert np.isfinite(est.ate)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("econml"),
    reason="econml not installed",
)
def test_double_ml_pvalue_in_range() -> None:
    rfm = _make_rfm(300)
    df, _, controls = prepare_causal_dataset(rfm)
    est = DoubleMLEstimator("onboarding_completed", cv_folds=3)
    est.fit(df, controls)
    p = est.get_ate_pvalue()
    assert 0 <= p <= 1


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("econml"),
    reason="econml not installed",
)
def test_cate_with_ci_ordering() -> None:
    rfm = _make_rfm(300)
    df, _, controls = prepare_causal_dataset(rfm)
    est = DoubleMLEstimator("fast_repeat_buyer", cv_folds=3)
    est.fit(df, controls)
    cate, lower, upper = est.estimate_cate_with_ci(df, controls)
    assert (upper >= lower).all(), "Upper CI must be >= lower CI"