"""Unit tests for CATE heterogeneity analysis."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backend.ml.causal_heterogeneous import (
    compute_heterogeneity_report,
    find_high_leverage_customers,
)


def _make_cate_results(n: int = 100) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "onboarding_completed":      rng.normal(200, 100, n),
        "fast_repeat_buyer":         rng.normal(150, 80,  n),
        "high_value_first_purchase": rng.normal(300, 150, n),
    }


def _make_rfm(n: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "customer_id":   [f"C{i:04d}" for i in range(n)],
        "monetary_avg":  rng.uniform(10, 500, n).tolist(),
        "cohort_month":  ["2011-01"] * n,
        "actual_ltv_12m": rng.uniform(0, 2000, n).tolist(),
    })


def test_heterogeneity_report_columns() -> None:
    cate = _make_cate_results(100)
    rfm  = _make_rfm(100)
    report = compute_heterogeneity_report(cate, rfm)
    assert "treatment_name" in report.columns
    assert "ate" in report.columns
    assert "heterogeneity_index" in report.columns
    assert "pct_positive_cate" in report.columns


def test_heterogeneity_report_length() -> None:
    cate   = _make_cate_results(100)
    rfm    = _make_rfm(100)
    report = compute_heterogeneity_report(cate, rfm)
    assert len(report) == len(cate)


def test_heterogeneity_report_sorted_by_ate() -> None:
    cate   = _make_cate_results(100)
    rfm    = _make_rfm(100)
    report = compute_heterogeneity_report(cate, rfm)
    ates   = report["ate"].to_list()
    assert ates == sorted(ates, reverse=True)


def test_find_high_leverage_returns_polars() -> None:
    cate  = _make_cate_results(100)
    ids   = [f"C{i:04d}" for i in range(100)]
    result = find_high_leverage_customers(cate, ids, min_total_uplift=0, top_n=10)
    assert isinstance(result, pl.DataFrame)
    assert len(result) <= 10


def test_find_high_leverage_sorted() -> None:
    cate  = _make_cate_results(100)
    ids   = [f"C{i:04d}" for i in range(100)]
    result = find_high_leverage_customers(cate, ids, min_total_uplift=0, top_n=50)
    uplifts = result["total_uplift"].to_list()
    assert uplifts == sorted(uplifts, reverse=True)


def test_find_high_leverage_min_filter() -> None:
    cate  = _make_cate_results(100)
    ids   = [f"C{i:04d}" for i in range(100)]
    result = find_high_leverage_customers(cate, ids, min_total_uplift=1e9, top_n=100)
    assert len(result) == 0


def test_find_high_leverage_has_cate_columns() -> None:
    cate  = _make_cate_results(100)
    ids   = [f"C{i:04d}" for i in range(100)]
    result = find_high_leverage_customers(cate, ids, min_total_uplift=0, top_n=10)
    for t in cate:
        assert f"cate_{t}" in result.columns