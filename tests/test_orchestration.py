"""Unit tests for Dagster assets (lightweight — no full pipeline run)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from unittest.mock import MagicMock, patch


def _make_rfm(n: int = 100) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    t   = rng.integers(180, 730, n)
    return pl.DataFrame({
        "customer_id":              [f"C{i:05d}" for i in range(n)],
        "frequency":                rng.integers(0, 10, n).astype(int).tolist(),
        "recency_days":             rng.uniform(0, 300, n).tolist(),
        "t_days":                   t.astype(int).tolist(),
        "monetary_avg":             rng.uniform(10, 500, n).tolist(),
        "monetary_total":           rng.uniform(50, 5000, n).tolist(),
        "monetary_std":             rng.uniform(0, 50, n).tolist(),
        "purchase_variance":        rng.uniform(0, 10000, n).tolist(),
        "orders_count":             rng.integers(1, 10, n).astype(int).tolist(),
        "avg_days_between_orders":  rng.uniform(10, 100, n).tolist(),
        "unique_categories":        rng.integers(1, 5, n).astype(int).tolist(),
        "unique_products":          rng.integers(1, 15, n).astype(int).tolist(),
        "cohort_month":             ["2011-01"] * n,
        "observation_end_date":     ["2011-06-30"] * n,
        "days_to_second_purchase":  rng.integers(1, 365, n).astype(int).tolist(),
        "first_purchase_amount":    rng.uniform(5, 200, n).tolist(),
        "multi_country":            [False] * n,
        "actual_ltv_12m":           rng.uniform(0, 2000, n).tolist(),
    })


def test_cleaned_transactions_asset_runs() -> None:
    """Verify the clean_transactions function used by the asset works."""
    from backend.features.rfm import clean_transactions, assign_product_categories
    import polars as pl
    from datetime import datetime, timezone

    raw = pl.DataFrame({
        "invoice_no":  ["A001", "C002", "A003"],   # C002 is a cancellation
        "stock_code":  ["21045", "22500", "84970"],
        "description": ["ITEM", "ITEM", "ITEM"],
        "quantity":    [2, -1, 3],
        "invoice_date":[datetime(2011, 1, 1, tzinfo=timezone.utc)] * 3,
        "unit_price":  [5.0, 8.0, 3.0],
        "customer_id": ["C001", "C002", None],
        "country":     ["UK", "UK", "UK"],
    })

    cleaned = clean_transactions(raw)
    # Should only keep row 0 (row 1 is cancellation, row 2 has null customer)
    assert len(cleaned) == 1
    assert cleaned["invoice_no"][0] == "A001"


def test_schedules_defined() -> None:
    """Verify all schedule objects are importable."""
    from orchestration.schedules import (
        monthly_retraining_schedule,
        weekly_drift_check_schedule,
        daily_data_refresh_schedule,
    )
    assert monthly_retraining_schedule is not None
    assert weekly_drift_check_schedule is not None
    assert daily_data_refresh_schedule is not None


def test_dagster_defs_importable() -> None:
    """Verify Dagster definitions can be imported without error."""
    try:
        from orchestration.dagster_assets import defs
        assert defs is not None
    except ImportError as e:
        pytest.skip(f"Dagster not installed: {e}")


def test_resources_instantiable() -> None:
    """Verify resource classes can be instantiated."""
    from orchestration.resources import (
        SupabaseResource,
        WandbResource,
        ModelStorageResource,
    )
    s = SupabaseResource(database_url="", service_role_key="", supabase_url="")
    w = WandbResource(api_key="", project="test")
    m = ModelStorageResource(models_dir="./models")
    assert s is not None
    assert w is not None
    assert m is not None