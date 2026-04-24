"""Unit tests for CDNOW benchmark validation."""

from __future__ import annotations

import pytest


def test_cdnow_loads() -> None:
    """CDNOW dataset should load from lifetimes library without error."""
    from backend.ml.cdnow_validation import load_cdnow_as_polars
    cal, hold = load_cdnow_as_polars()
    assert len(cal) > 2000
    assert len(hold) > 2000
    assert "frequency" in cal.columns
    assert "recency_days" in cal.columns
    assert "t_days" in cal.columns
    assert "monetary_avg" in cal.columns


def test_cdnow_benchmark_runs() -> None:
    """Benchmark should run without crashing and return expected keys."""
    from backend.ml.cdnow_validation import run_cdnow_benchmark
    results = run_cdnow_benchmark(penalizer=0.001)
    assert "fitted_params" in results
    assert "metrics" in results
    assert "benchmark_pass" in results
    # Params should be positive
    for v in results["fitted_params"].values():
        assert v > 0


def test_cdnow_r2_above_threshold() -> None:
    """BG/NBD should achieve R² > 0.75 on CDNOW (relaxed for CI speed)."""
    from backend.ml.cdnow_validation import run_cdnow_benchmark
    results = run_cdnow_benchmark(penalizer=0.001)
    assert results["metrics"]["r2_frequency"] > 0.75