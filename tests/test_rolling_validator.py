"""Unit tests for RollingValidator (mocked DB)."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.monitoring.rolling_validator import RollingValidator


def _make_mock_db(n: int = 200) -> MagicMock:
    """Return a mock DB client with synthetic prediction/actual data."""
    rng = np.random.default_rng(42)
    true_ltv = rng.exponential(scale=500, size=n).clip(0)
    pred_ltv = true_ltv * rng.uniform(0.8, 1.2, n)

    mock_rows = [
        {
            "customer_id":        f"C{i:05d}",
            "predicted_ltv_12m":  float(pred_ltv[i]),
            "actual_ltv_12m":     float(true_ltv[i]),
        }
        for i in range(n)
    ]

    db = MagicMock()
    db.execute_sql.return_value = mock_rows
    db.bulk_upsert.return_value = n
    return db


def test_rolling_validator_returns_metrics() -> None:
    db        = _make_mock_db(200)
    validator = RollingValidator(db_client=db)
    result    = validator.run()

    assert "mae_ltv_12m"       in result
    assert "gini_coefficient"  in result
    assert "top_decile_lift"   in result
    assert "calibration_error" in result
    assert "n_customers"       in result


def test_rolling_validator_mae_non_negative() -> None:
    db        = _make_mock_db(100)
    validator = RollingValidator(db_client=db)
    result    = validator.run()
    assert result["mae_ltv_12m"] >= 0


def test_rolling_validator_gini_in_range() -> None:
    db        = _make_mock_db(200)
    validator = RollingValidator(db_client=db)
    result    = validator.run()
    assert -1 <= result["gini_coefficient"] <= 1


def test_rolling_validator_no_data() -> None:
    db        = MagicMock()
    db.execute_sql.return_value = []
    validator = RollingValidator(db_client=db)
    result    = validator.run()
    assert result == {}


def test_rolling_validator_n_customers() -> None:
    n         = 150
    db        = _make_mock_db(n)
    validator = RollingValidator(db_client=db)
    result    = validator.run()
    assert result["n_customers"] == n