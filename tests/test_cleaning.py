"""Unit tests for data cleaning and feature engineering."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from backend.features.rfm import (
    assign_amount_buckets,
    assign_product_categories,
    clean_transactions,
)


def make_sample_df() -> pl.DataFrame:
    return pl.DataFrame({
        "invoice_no":   ["536365", "C536366", "536367", "536368", "536369"],
        "stock_code":   ["85123A", "22633",   "21730",  "84879",  "21232"],
        "description":  ["WHITE ITEM", "THROW", "GLASS", "WRAP",  "CANDLE"],
        "quantity":     [6,           -2,       12,       1,       0],
        "invoice_date": [
            datetime(2010, 12, 1,  8, 26, tzinfo=timezone.utc),
            datetime(2010, 12, 1,  9, 0,  tzinfo=timezone.utc),
            datetime(2010, 12, 1, 10, 0,  tzinfo=timezone.utc),
            datetime(2010, 12, 1, 11, 0,  tzinfo=timezone.utc),
            datetime(2010, 12, 1, 12, 0,  tzinfo=timezone.utc),
        ],
        "unit_price":   [2.55, 6.35, 4.25, 0.0, 1.25],
        "customer_id":  ["17850", "17850", None, "17851", "17852"],
        "country":      ["United Kingdom"] * 5,
    })


def test_clean_removes_cancellations() -> None:
    df = make_sample_df()
    cleaned = clean_transactions(df)
    assert not any(cleaned["invoice_no"].str.starts_with("C"))


def test_clean_removes_null_customer() -> None:
    df = make_sample_df()
    cleaned = clean_transactions(df)
    assert cleaned["customer_id"].null_count() == 0


def test_clean_removes_zero_quantity() -> None:
    df = make_sample_df()
    cleaned = clean_transactions(df)
    assert (cleaned["quantity"] > 0).all()


def test_clean_removes_zero_price() -> None:
    df = make_sample_df()
    cleaned = clean_transactions(df)
    assert (cleaned["unit_price"] > 0).all()


def test_clean_retains_valid_row() -> None:
    df = make_sample_df()
    cleaned = clean_transactions(df)
    # Only invoice 536365 (qty=6, price=2.55, customer=17850) should survive
    assert "536365" in cleaned["invoice_no"].to_list()


def test_assign_product_categories_known() -> None:
    df = pl.DataFrame({
        "stock_code": ["21045", "22500", "84970", "POSTAGE"],
        "customer_id": ["1", "2", "3", "4"],
        "invoice_no": ["A", "B", "C", "D"],
        "quantity": [1, 1, 1, 1],
        "unit_price": [1.0, 1.0, 1.0, 1.0],
        "invoice_date": [datetime(2011, 1, 1, tzinfo=timezone.utc)] * 4,
        "country": ["UK"] * 4,
    })
    with_cats = assign_product_categories(df)
    assert "product_category" in with_cats.columns
    assert with_cats.filter(pl.col("stock_code") == "22500")["product_category"][0] == "kitchenware"


def test_assign_amount_buckets_five_bins() -> None:
    n = 100
    df = pl.DataFrame({
        "invoice_no":   [str(i) for i in range(n)],
        "customer_id":  [str(i) for i in range(n)],
        "quantity":     [1] * n,
        "unit_price":   list(range(1, n + 1, 1)),
        "invoice_date": [datetime(2011, 1, 1, tzinfo=timezone.utc)] * n,
        "country":      ["UK"] * n,
        "line_total":   list(range(1, n + 1, 1)),
        "product_category": ["homewares"] * n,
    })
    bucketed = assign_amount_buckets(df)
    assert set(bucketed["amount_bucket"].to_list()).issubset({1, 2, 3, 4, 5})