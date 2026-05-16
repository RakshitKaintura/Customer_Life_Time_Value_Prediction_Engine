"""Unit tests for drift detection."""

from __future__ import annotations

import numpy as np
import pytest

from backend.monitoring.drift import (
    compute_psi,
    compute_ks_test,
    compute_feature_drift,
)


def test_psi_identical_distributions() -> None:
    rng = np.random.default_rng(42)
    x   = rng.normal(100, 20, 1000)
    psi = compute_psi(x, x)
    assert psi < 0.05, f"PSI should be ~0 for identical distributions, got {psi}"


def test_psi_very_different_distributions() -> None:
    rng      = np.random.default_rng(42)
    baseline = rng.normal(100, 20, 1000)
    current  = rng.normal(300, 20, 1000)   # massively shifted
    psi      = compute_psi(baseline, current)
    assert psi > 0.25, f"PSI should be high for very different distributions, got {psi}"


def test_psi_moderate_shift() -> None:
    rng      = np.random.default_rng(42)
    baseline = rng.normal(100, 20, 1000)
    current  = rng.normal(115, 22, 1000)   # 15% shift
    psi      = compute_psi(baseline, current)
    assert 0.0 < psi < 1.0


def test_psi_non_negative() -> None:
    rng = np.random.default_rng(42)
    for _ in range(5):
        a = rng.exponential(50, 500)
        b = rng.exponential(55, 500)
        assert compute_psi(a, b) >= 0


def test_ks_test_same_distribution() -> None:
    rng        = np.random.default_rng(42)
    x          = rng.normal(100, 20, 1000)
    y          = rng.normal(100, 20, 1000)
    ks, pvalue = compute_ks_test(x, y)
    assert pvalue > 0.05, "Same distribution should not be significant"
    assert 0 <= ks <= 1


def test_ks_test_different_distributions() -> None:
    rng        = np.random.default_rng(42)
    x          = rng.normal(100, 20, 1000)
    y          = rng.normal(200, 20, 1000)
    ks, pvalue = compute_ks_test(x, y)
    assert pvalue < 0.05, "Different distributions should be significant"
    assert ks > 0.5


def test_feature_drift_returns_all_keys() -> None:
    rng      = np.random.default_rng(42)
    baseline = rng.normal(50, 10, 500)
    current  = rng.normal(55, 12, 500)
    result   = compute_feature_drift(baseline, current)

    required_keys = [
        "psi_score", "ks_statistic", "baseline_mean", "baseline_std",
        "current_mean", "current_std", "mean_shift_pct", "is_drifted",
    ]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


def test_feature_drift_no_drift_flag() -> None:
    rng      = np.random.default_rng(42)
    baseline = rng.normal(100, 10, 1000)
    current  = rng.normal(100, 10, 1000)   # same
    result   = compute_feature_drift(baseline, current, threshold=0.20)
    assert result["is_drifted"] is False


def test_feature_drift_with_drift_flag() -> None:
    rng      = np.random.default_rng(42)
    baseline = rng.normal(100, 10, 1000)
    current  = rng.normal(200, 10, 1000)   # very different
    result   = compute_feature_drift(baseline, current, threshold=0.20)
    assert result["is_drifted"] is True


def test_mean_shift_pct_positive_shift() -> None:
    rng      = np.random.default_rng(42)
    baseline = rng.normal(100, 5, 500)
    current  = rng.normal(120, 5, 500)   # 20% shift
    result   = compute_feature_drift(baseline, current)
    assert abs(result["mean_shift_pct"] - 20.0) < 5.0


def test_psi_minimum_data() -> None:
    """PSI should handle small arrays without error."""
    rng      = np.random.default_rng(42)
    baseline = rng.normal(100, 10, 20)
    current  = rng.normal(110, 10, 20)
    psi      = compute_psi(baseline, current, n_bins=5)
    assert np.isfinite(psi)