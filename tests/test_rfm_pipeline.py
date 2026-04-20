"""Unit tests for the RFMPipeline class."""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta

import polars as pl
import pytest

from backend.features.rfm import RFMPipeline, make_calibration_holdout_split


def _make_transactions(n_customers: int = 10, n_invoices_each: int = 3) -> pl.DataFrame:
    """Create a synthetic transaction dataset for testing."""
    rows = []
    base_date = datetime(2010, 12, 1, tzinfo=timezone.utc)
    for cust_i in range(n_customers):
        customer_id = f"C{cust_i:04d}"
        for inv_i in range(n_invoices_each):
            rows.append({
                "invoice_no":       f"INV{cust_i:04d}{inv_i:02d}",
                "stock_code":       "21045",
                "description":      "TEST ITEM",
                "quantity":         3,
                "invoice_date":     base_date + timedelta(days=inv_i * 30 + cust_i),
                "unit_price":       10.0 + inv_i,
                "line_total":       30.0 + inv_i * 10,
                "customer_id":      customer_id,
                "country":          "United Kingdom",
                "product_category": "homewares",
                "amount_bucket":    2,
            })

    return pl.DataFrame(rows)


def test_rfm_one_row_per_customer() -> None:
    df = _make_transactions(n_customers=10)
    pipeline = RFMPipeline(df, observation_end_date=date(2011, 6, 30))
    rfm = pipeline.compute()
    assert len(rfm) == 10
    assert rfm["customer_id"].n_unique() == 10


def test_rfm_frequency_is_non_negative() -> None:
    df = _make_transactions()
    pipeline = RFMPipeline(df, observation_end_date=date(2011, 6, 30))
    rfm = pipeline.compute()
    assert (rfm["frequency"] >= 0).all()


def test_rfm_t_days_positive() -> None:
    df = _make_transactions()
    pipeline = RFMPipeline(df, observation_end_date=date(2011, 6, 30))
    rfm = pipeline.compute()
    assert (rfm["t_days"] >= 0).all()


def test_rfm_recency_leq_t_days() -> None:
    """Recency must be ≤ T (customer age) — BG/NBD constraint."""
    df = _make_transactions()
    pipeline = RFMPipeline(df, observation_end_date=date(2011, 6, 30))
    rfm = pipeline.compute()
    assert (rfm["recency_days"] <= rfm["t_days"].cast(pl.Float64)).all()


def test_calibration_holdout_split_no_overlap() -> None:
    df = _make_transactions(n_customers=20, n_invoices_each=5)
    cal, hold, obs_end, hold_end = make_calibration_holdout_split(
        df, observation_months=3, holdout_months=3
    )
    cal_dates = set(cal["invoice_date"].cast(pl.Date).to_list())
    hold_dates = set(hold["invoice_date"].cast(pl.Date).to_list())
    # Calibration must be strictly before observation_end
    assert all(d <= obs_end for d in cal_dates if d is not None)
    # Holdout must be strictly after observation_end
    assert all(d > obs_end for d in hold_dates if d is not None)


def test_ltv_labels_non_negative() -> None:
    df = _make_transactions(n_customers=5, n_invoices_each=6)
    cal, hold, obs_end, _ = make_calibration_holdout_split(
        df, observation_months=3, holdout_months=3
    )
    pipeline = RFMPipeline(cal, observation_end_date=obs_end)
    rfm = pipeline.compute()
    labelled = pipeline.compute_ltv_labels(hold, rfm, horizon_months=12)
    assert "actual_ltv_12m" in labelled.columns
    assert (labelled["actual_ltv_12m"] >= 0).all()