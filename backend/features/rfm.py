"""
RFM Feature Engineering Pipeline — Polars implementation.

Computes:
  - Recency, Frequency, Monetary (BG/NBD + Gamma-Gamma inputs)
  - Calibration / holdout split
  - Ground-truth LTV labels (filled after holdout window elapses)
  - Persists results to Supabase

Polars lazy API is used throughout for memory efficiency on 500K+ rows.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from backend.db.models import RFMFeatures

if TYPE_CHECKING:
    from backend.db.supabase_client import SupabaseClient


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MARGIN = 0.20             # assumed gross margin for LTV calculation
ANNUAL_DISCOUNT_RATE = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply data quality rules to raw UCI Online Retail data.

    Rules:
      - Remove rows with null CustomerID
      - Remove returns / cancellations (InvoiceNo starts with 'C')
      - Remove zero or negative Quantity
      - Remove zero or negative UnitPrice
      - Remove clearly erroneous UnitPrice > 10_000
      - Cast InvoiceDate to Datetime
    """
    logger.info("Cleaning transactions — {} raw rows", len(df))

    cleaned = (
        df.lazy()
        .filter(pl.col("customer_id").is_not_null())
        .filter(~pl.col("invoice_no").str.starts_with("C"))  # cancellations
        .filter(pl.col("quantity") > 0)
        .filter(pl.col("unit_price") > 0)
        .filter(pl.col("unit_price") < 10_000)
        .with_columns(
            # Normalise column types
            pl.col("customer_id").cast(pl.Utf8).str.strip_chars(),
            pl.col("invoice_no").str.strip_chars(),
            pl.col("stock_code").str.strip_chars().str.to_uppercase(),
            pl.col("description").str.strip_chars(),
            pl.col("country").str.strip_chars(),
            # Computed columns
            (pl.col("quantity") * pl.col("unit_price")).alias("line_total"),
        )
        .collect()
    )

    logger.info(
        "After cleaning: {} rows ({:.1f}% kept)",
        len(cleaned),
        100 * len(cleaned) / max(len(df), 1),
    )
    return cleaned


def assign_product_categories(df: pl.DataFrame) -> pl.DataFrame:
    """
    Heuristic category assignment from UCI stock code prefixes.
    Polars expression-based — no UDF overhead.
    """
    return df.with_columns(
        pl.when(pl.col("stock_code").str.starts_with("20")).then(pl.lit("gift_wrap"))
        .when(pl.col("stock_code").str.starts_with("21")).then(pl.lit("homewares"))
        .when(pl.col("stock_code").str.starts_with("22")).then(pl.lit("kitchenware"))
        .when(pl.col("stock_code").str.starts_with("23")).then(pl.lit("bags_cases"))
        .when(pl.col("stock_code").str.starts_with("47")).then(pl.lit("stationery"))
        .when(pl.col("stock_code").str.starts_with("48")).then(pl.lit("seasonal"))
        .when(pl.col("stock_code").str.starts_with("71")).then(pl.lit("art_craft"))
        .when(pl.col("stock_code").str.starts_with("84")).then(pl.lit("decorative"))
        .when(pl.col("stock_code").str.starts_with("85")).then(pl.lit("novelty"))
        .when(pl.col("stock_code").str.starts_with("POST")).then(pl.lit("postage"))
        .otherwise(pl.lit("other"))
        .alias("product_category")
    )


def assign_amount_buckets(df: pl.DataFrame, n_buckets: int = 5) -> pl.DataFrame:
    """
    Assign each transaction's line_total to a quantile bucket (1 = lowest).
    Used as a Transformer token feature.
    """
    quantiles = df["line_total"].quantile([i / n_buckets for i in range(1, n_buckets)])
    cuts = [float(quantiles[i]) for i in range(len(quantiles))]

    return df.with_columns(
        pl.when(pl.col("line_total") <= cuts[0]).then(1)
        .when(pl.col("line_total") <= cuts[1]).then(2)
        .when(pl.col("line_total") <= cuts[2]).then(3)
        .when(pl.col("line_total") <= cuts[3]).then(4)
        .otherwise(5)
        .cast(pl.Int8)
        .alias("amount_bucket")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Calibration / Holdout split
# ──────────────────────────────────────────────────────────────────────────────

def make_calibration_holdout_split(
    df: pl.DataFrame,
    observation_months: int = 6,
    holdout_months: int = 6,
    date_col: str = "invoice_date",
) -> tuple[pl.DataFrame, pl.DataFrame, date, date]:
    """
    Split transactions into calibration (observation) and holdout windows.

    Strategy:
      - dataset_start  = earliest invoice date
      - obs_end        = dataset_start + observation_months
      - holdout_end    = obs_end + holdout_months

    Returns:
        calibration_df, holdout_df, obs_end_date, holdout_end_date
    """
    dates = df[date_col].cast(pl.Date)
    dataset_start: date = dates.min()

    obs_end = dataset_start + timedelta(days=observation_months * 30)
    holdout_end = obs_end + timedelta(days=holdout_months * 30)

    calibration_df = df.filter(pl.col(date_col).cast(pl.Date) <= obs_end)
    holdout_df = df.filter(
        (pl.col(date_col).cast(pl.Date) > obs_end)
        & (pl.col(date_col).cast(pl.Date) <= holdout_end)
    )

    logger.info(
        "Split → calibration {} rows (≤{}), holdout {} rows ({} – {})",
        len(calibration_df), obs_end,
        len(holdout_df), obs_end, holdout_end,
    )
    return calibration_df, holdout_df, obs_end, holdout_end


# ──────────────────────────────────────────────────────────────────────────────
# RFM Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class RFMPipeline:
    """
    Full RFM feature engineering pipeline using Polars lazy evaluation.

    Usage:
        pipeline = RFMPipeline(calibration_df, observation_end_date)
        rfm = pipeline.compute()
        pipeline.save(rfm, db_client)
    """

    def __init__(
        self,
        transactions: pl.DataFrame,
        observation_end_date: date,
        pipeline_run_id: str | None = None,
    ) -> None:
        self.transactions = transactions
        self.observation_end = observation_end_date
        self.run_id = pipeline_run_id or str(uuid.uuid4())[:8]

    def compute(self) -> pl.DataFrame:
        """Run the full RFM computation. Returns one row per customer."""
        logger.info(
            "Computing RFM for {} transactions up to {}",
            len(self.transactions), self.observation_end,
        )

        # Step 1: per-invoice totals
        invoice_totals = (
            self.transactions.lazy()
            .filter(pl.col("customer_id").is_not_null())
            .with_columns(
                pl.col("invoice_date").cast(pl.Date).alias("purchase_date")
            )
            .group_by(["customer_id", "invoice_no", "purchase_date"])
            .agg(
                pl.sum("line_total").alias("invoice_total"),
                pl.sum("quantity").alias("invoice_items"),
                pl.first("product_category").alias("category"),
                pl.first("country").alias("country"),
            )
        )

        # Step 2: per-customer aggregation
        customer_agg = (
            invoice_totals
            .group_by("customer_id")
            .agg(
                # BG/NBD inputs
                (pl.col("purchase_date").n_unique() - 1)
                    .cast(pl.Int32).alias("frequency"),
                pl.col("invoice_total").mean().alias("monetary_avg"),
                pl.col("invoice_total").std().alias("monetary_std"),
                pl.col("invoice_total").sum().alias("monetary_total"),
                pl.col("invoice_total").var().alias("purchase_variance"),
                # Date range
                pl.col("purchase_date").min().alias("first_purchase_date"),
                pl.col("purchase_date").max().alias("last_purchase_date"),
                # Counts
                pl.col("invoice_no").n_unique().alias("unique_invoices"),
                pl.col("invoice_items").mean().alias("avg_items_per_order"),
                # Category
                pl.col("category").first().alias("first_purchase_category"),
                pl.col("invoice_total").first().alias("first_purchase_amount"),
                pl.col("category").n_unique().alias("unique_categories"),
                # Multi-country flag
                (pl.col("country").n_unique() > 1).alias("multi_country"),
            )
        )

        # Step 3: compute recency_days and t_days relative to observation_end
        obs_end_lit = pl.lit(self.observation_end)

        rfm = (
            customer_agg
            .with_columns(
                # recency = days from first to last purchase
                (pl.col("last_purchase_date") - pl.col("first_purchase_date"))
                    .dt.total_days().cast(pl.Float64).alias("recency_days"),
                # T = days from first purchase to observation end
                (obs_end_lit - pl.col("first_purchase_date"))
                    .dt.total_days().cast(pl.Int32).alias("t_days"),
                # cohort_month
                pl.col("first_purchase_date")
                    .dt.to_string("%Y-%m").alias("cohort_month"),
                # observation_end metadata
                pl.lit(self.observation_end).alias("observation_end_date"),
                pl.lit(self.run_id).alias("pipeline_run_id"),
                pl.lit(len(self.transactions)).alias("_n_source_rows"),
            )
            # Re-alias orders_count
            .with_columns(
                pl.col("unique_invoices").alias("orders_count"),
            )
            .collect()
        )

        # Step 4: inter-purchase times (requires ordered data per customer)
        ipt = self._compute_inter_purchase_times()
        rfm = rfm.join(ipt, on="customer_id", how="left")

        # Step 5: days to second purchase
        d2s = self._compute_days_to_second_purchase()
        rfm = rfm.join(d2s, on="customer_id", how="left")

        # Step 6: unique product count
        uniq_products = (
            self.transactions.lazy()
            .group_by("customer_id")
            .agg(pl.col("stock_code").n_unique().alias("unique_products"))
            .collect()
        )
        rfm = rfm.join(uniq_products, on="customer_id", how="left")

        logger.info(
            "RFM computed for {} customers — obs_end={}",
            len(rfm), self.observation_end,
        )
        return rfm

    def _compute_inter_purchase_times(self) -> pl.DataFrame:
        """Per-customer average and std of days between purchase dates."""
        daily = (
            self.transactions.lazy()
            .with_columns(pl.col("invoice_date").cast(pl.Date).alias("purchase_date"))
            .group_by(["customer_id", "purchase_date"])
            .agg(pl.lit(1))   # deduplicate by day
            .sort(["customer_id", "purchase_date"])
        )

        # Shift within each customer group to compute lag
        with_lag = (
            daily
            .with_columns(
                pl.col("purchase_date")
                    .shift(1)
                    .over("customer_id")
                    .alias("prev_purchase_date")
            )
            .with_columns(
                (pl.col("purchase_date") - pl.col("prev_purchase_date"))
                    .dt.total_days()
                    .alias("days_between")
            )
            .filter(pl.col("days_between").is_not_null())
            .group_by("customer_id")
            .agg(
                pl.col("days_between").mean().alias("avg_days_between_orders"),
                pl.col("days_between").std().alias("std_days_between_orders"),
            )
            .collect()
        )
        return with_lag

    def _compute_days_to_second_purchase(self) -> pl.DataFrame:
        """Days between first and second purchase date per customer."""
        ranked = (
            self.transactions.lazy()
            .with_columns(pl.col("invoice_date").cast(pl.Date).alias("purchase_date"))
            .group_by(["customer_id", "purchase_date"])
            .agg(pl.lit(1))
            .sort(["customer_id", "purchase_date"])
            .with_columns(
                pl.col("purchase_date")
                    .rank(method="ordinal")
                    .over("customer_id")
                    .alias("purchase_rank")
            )
            .collect()
        )

        first = ranked.filter(pl.col("purchase_rank") == 1).select(
            ["customer_id", pl.col("purchase_date").alias("first_date")]
        )
        second = ranked.filter(pl.col("purchase_rank") == 2).select(
            ["customer_id", pl.col("purchase_date").alias("second_date")]
        )

        joined = (
            first.join(second, on="customer_id", how="left")
            .with_columns(
                (pl.col("second_date") - pl.col("first_date"))
                    .dt.total_days()
                    .cast(pl.Int32)
                    .alias("days_to_second_purchase")
            )
            .select(["customer_id", "days_to_second_purchase"])
        )
        return joined

    def compute_ltv_labels(
        self,
        holdout_df: pl.DataFrame,
        rfm_df: pl.DataFrame,
        horizon_months: int = 12,
    ) -> pl.DataFrame:
        """
        Compute actual LTV labels from the holdout period.

        Args:
            holdout_df:     transactions in the holdout window
            rfm_df:         existing RFM dataframe to attach labels to
            horizon_months: 12 | 24 | 36

        Returns rfm_df with `actual_ltv_{horizon}m` column added.
        """
        col_name = f"actual_ltv_{horizon_months}m"
        logger.info(
            "Computing actual LTV labels for {}m horizon ({} holdout rows)",
            horizon_months, len(holdout_df),
        )

        horizon_end = self.observation_end + timedelta(days=horizon_months * 30)

        actual = (
            holdout_df.lazy()
            .filter(pl.col("customer_id").is_not_null())
            .filter(pl.col("invoice_date").cast(pl.Date) > pl.lit(self.observation_end))
            .filter(pl.col("invoice_date").cast(pl.Date) <= pl.lit(horizon_end))
            .group_by("customer_id")
            .agg(
                (pl.col("quantity") * pl.col("unit_price"))
                    .sum()
                    .alias(col_name)
            )
            .collect()
        )

        # Left join — customers with no holdout activity get 0 LTV
        labelled = (
            rfm_df
            .join(actual, on="customer_id", how="left")
            .with_columns(
                pl.col(col_name).fill_null(0.0)
            )
        )
        logger.info(
            "{} label — {} / {} customers have >0 holdout revenue",
            col_name,
            (labelled[col_name] > 0).sum(),
            len(labelled),
        )
        return labelled

    def save(
        self,
        rfm_df: pl.DataFrame,
        db_client: "SupabaseClient",
        batch_size: int = 500,
    ) -> int:
        """Persist RFM features to Supabase rfm_features table."""
        logger.info("Saving {} RFM rows to Supabase…", len(rfm_df))

        # Convert Polars → list of dicts (handle Date serialisation)
        records = rfm_df.with_columns(
            pl.col("observation_end_date").cast(pl.Utf8),
            pl.col("first_purchase_date").cast(pl.Utf8),
            pl.col("last_purchase_date").cast(pl.Utf8),
        ).to_dicts()

        # Keep only columns that exist in the DB schema
        valid_cols = RFMFeatures.__table__.columns.keys()
        records = [
            {k: v for k, v in r.items() if k in valid_cols}
            for r in records
        ]

        inserted = db_client.bulk_upsert(
            table_name="rfm_features",
            records=records,
            conflict_columns=["customer_id", "observation_end_date"],
            batch_size=batch_size,
        )
        logger.info("Saved {} RFM rows", inserted)
        return inserted
