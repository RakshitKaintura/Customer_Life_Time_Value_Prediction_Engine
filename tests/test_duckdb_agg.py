"""Unit tests for DuckDBAggregator."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from backend.features.duckdb_agg import DuckDBAggregator


def _sample_transactions() -> pl.DataFrame:
    return pl.DataFrame({
        "invoice_no":       ["A001", "A002", "A003", "A004"],
        "stock_code":       ["21045", "22500", "21045", "22500"],
        "customer_id":      ["C001",  "C001",  "C002",  "C003"],
        "quantity":         [2, 3, 1, 4],
        "unit_price":       [5.0, 8.0, 5.0, 8.0],
        "invoice_date":     [
            datetime(2011, 1, 10, tzinfo=timezone.utc),
            datetime(2011, 2, 15, tzinfo=timezone.utc),
            datetime(2011, 1, 20, tzinfo=timezone.utc),
            datetime(2011, 3, 5,  tzinfo=timezone.utc),
        ],
        "country":          ["UK", "UK", "FR", "UK"],
        "product_category": ["homewares", "kitchenware", "homewares", "kitchenware"],
    })


def test_basic_stats() -> None:
    df = _sample_transactions()
    with DuckDBAggregator() as duck:
        duck.register_polars("transactions", df)
        stats = duck.agg_basic_stats("transactions")
    assert stats["unique_customers"][0] == 3
    assert stats["total_rows"][0] == 4


def test_customer_totals() -> None:
    df = _sample_transactions()
    with DuckDBAggregator() as duck:
        duck.register_polars("transactions", df)
        totals = duck.agg_customer_totals("transactions")
    assert len(totals) == 3
    c001 = totals.filter(pl.col("customer_id") == "C001")
    assert c001["total_orders"][0] == 2


def test_rfm_base_frequency() -> None:
    df = _sample_transactions()
    with DuckDBAggregator() as duck:
        duck.register_polars("transactions", df)
        rfm = duck.agg_rfm_base("transactions", observation_end="2011-12-31")
    # C001 has 2 unique purchase dates → frequency = 1
    c001 = rfm.filter(pl.col("customer_id") == "C001")
    assert c001["frequency"][0] == 1


def test_context_manager_closes() -> None:
    agg = DuckDBAggregator()
    with agg:
        assert agg._conn is not None
    assert agg._conn is None