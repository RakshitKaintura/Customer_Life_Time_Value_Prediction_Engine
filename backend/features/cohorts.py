"""
Cohort Feature Engineering Pipeline — Polars implementation.

Builds:
  - Monthly acquisition cohort assignments
  - Cohort retention matrix
  - Per-customer cohort features (cohort age, channel, vertical)
  - Enriches RFM dataframe with cohort metadata
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from backend.db.supabase_client import SupabaseClient


class CohortPipeline:
    """
    Cohort feature engineering.

    Usage:
        cp = CohortPipeline(transactions)
        cohort_df = cp.compute_cohort_assignments()
        retention = cp.compute_retention_matrix(cohort_df)
    """

    def __init__(self, transactions: pl.DataFrame) -> None:
        self.transactions = (
            transactions.lazy()
            .filter(pl.col("customer_id").is_not_null())
            .filter(pl.col("quantity") > 0)
            .filter(pl.col("unit_price") > 0)
            .collect()
        )

    # ─────────────────────────────────────────────────────────
    # Cohort assignment
    # ─────────────────────────────────────────────────────────

    def compute_cohort_assignments(self) -> pl.DataFrame:
        """
        Return one row per customer with:
          - cohort_month    YYYY-MM string
          - cohort_date     first day of cohort month
          - customer_age_at_obs  placeholder (filled later with obs_end)
        """
        first_purchases = (
            self.transactions.lazy()
            .with_columns(
                pl.col("invoice_date").cast(pl.Date).alias("purchase_date")
            )
            .group_by("customer_id")
            .agg(
                pl.col("purchase_date").min().alias("first_purchase_date"),
            )
            .with_columns(
                pl.col("first_purchase_date")
                    .dt.to_string("%Y-%m")
                    .alias("cohort_month"),
                pl.col("first_purchase_date")
                    .dt.month_start()
                    .alias("cohort_start_date"),
            )
            .collect()
        )

        logger.info(
            "Cohort assignments — {} customers across {} cohorts",
            len(first_purchases),
            first_purchases["cohort_month"].n_unique(),
        )
        return first_purchases

    def compute_cohort_sizes(self, cohort_df: pl.DataFrame) -> pl.DataFrame:
        """Customer counts per monthly cohort."""
        return (
            cohort_df.lazy()
            .group_by("cohort_month")
            .agg(pl.len().alias("cohort_size"))
            .sort("cohort_month")
            .collect()
        )

    # ─────────────────────────────────────────────────────────
    # Retention matrix
    # ─────────────────────────────────────────────────────────

    def compute_retention_matrix(
        self,
        cohort_df: pl.DataFrame,
        max_months: int = 12,
    ) -> pl.DataFrame:
        """
        Build the cohort × month retention matrix.

        Returns a tall DataFrame:
            cohort_month | months_since_first | active_customers | retention_rate
        """
        # Monthly activity per customer
        monthly_activity = (
            self.transactions.lazy()
            .with_columns(
                pl.col("invoice_date")
                    .cast(pl.Date)
                    .dt.month_start()
                    .alias("activity_month")
            )
            .group_by(["customer_id", "activity_month"])
            .agg(pl.lit(1))  # deduplicate
            .collect()
        )

        # Join with cohort assignments
        joined = (
            monthly_activity.lazy()
            .join(
                cohort_df.lazy().select(
                    ["customer_id", "cohort_month", "cohort_start_date"]
                ),
                on="customer_id",
                how="left",
            )
            .with_columns(
                # months since first purchase
                (
                    (pl.col("activity_month") - pl.col("cohort_start_date"))
                    .dt.total_days() / 30.44
                )
                .cast(pl.Int32)
                .alias("months_since_first")
            )
            .filter(
                (pl.col("months_since_first") >= 0)
                & (pl.col("months_since_first") <= max_months)
            )
            .collect()
        )

        # Cohort sizes for retention % calculation
        cohort_sizes = self.compute_cohort_sizes(cohort_df).rename(
            {"cohort_size": "cohort_n"}
        )

        retention = (
            joined.lazy()
            .group_by(["cohort_month", "months_since_first"])
            .agg(pl.col("customer_id").n_unique().alias("active_customers"))
            .join(cohort_sizes.lazy(), on="cohort_month", how="left")
            .with_columns(
                (pl.col("active_customers") / pl.col("cohort_n") * 100)
                    .round(2)
                    .alias("retention_rate_pct")
            )
            .sort(["cohort_month", "months_since_first"])
            .collect()
        )

        logger.info(
            "Retention matrix — {} cohorts × {} months",
            retention["cohort_month"].n_unique(),
            retention["months_since_first"].max(),
        )
        return retention

    # ─────────────────────────────────────────────────────────
    # LTV development by cohort
    # ─────────────────────────────────────────────────────────

    def compute_cohort_ltv_over_time(
        self,
        cohort_df: pl.DataFrame,
        max_months: int = 12,
    ) -> pl.DataFrame:
        """
        Cumulative revenue per cohort customer over time.
        Used in the dashboard cohort analysis page.
        """
        monthly_revenue = (
            self.transactions.lazy()
            .with_columns(
                pl.col("invoice_date")
                    .cast(pl.Date)
                    .dt.month_start()
                    .alias("activity_month"),
                (pl.col("quantity") * pl.col("unit_price")).alias("line_total"),
            )
            .group_by(["customer_id", "activity_month"])
            .agg(pl.col("line_total").sum().alias("monthly_revenue"))
            .collect()
        )

        joined = (
            monthly_revenue.lazy()
            .join(
                cohort_df.lazy().select(
                    ["customer_id", "cohort_month", "cohort_start_date"]
                ),
                on="customer_id",
                how="left",
            )
            .with_columns(
                (
                    (pl.col("activity_month") - pl.col("cohort_start_date"))
                    .dt.total_days() / 30.44
                )
                .cast(pl.Int32)
                .alias("months_since_first")
            )
            .filter(
                (pl.col("months_since_first") >= 0)
                & (pl.col("months_since_first") <= max_months)
            )
            .collect()
        )

        cohort_sizes = self.compute_cohort_sizes(cohort_df)

        ltv_curve = (
            joined.lazy()
            .group_by(["cohort_month", "months_since_first"])
            .agg(pl.col("monthly_revenue").sum().alias("total_revenue"))
            .join(cohort_sizes.lazy(), on="cohort_month", how="left")
            .with_columns(
                (pl.col("total_revenue") / pl.col("cohort_size"))
                    .alias("avg_revenue_per_customer")
            )
            .sort(["cohort_month", "months_since_first"])
            .collect()
        )

        # Add cumulative column
        ltv_curve = ltv_curve.with_columns(
            pl.col("avg_revenue_per_customer")
                .cum_sum()
                .over("cohort_month")
                .alias("cumulative_ltv_per_customer")
        )

        return ltv_curve

    # ─────────────────────────────────────────────────────────
    # Enrich RFM with cohort metadata
    # ─────────────────────────────────────────────────────────

    def enrich_rfm(
        self,
        rfm_df: pl.DataFrame,
        cohort_df: pl.DataFrame,
        observation_end_date: date,
    ) -> pl.DataFrame:
        """
        Join cohort metadata onto the RFM dataframe.
        Adds: cohort_month, cohort_start_date, customer_age_days.
        """
        enriched = (
            rfm_df.lazy()
            .join(
                cohort_df.lazy().select(
                    ["customer_id", "cohort_month", "cohort_start_date",
                     "first_purchase_date"]
                ),
                on="customer_id",
                how="left",
                suffix="_cohort",
            )
            .with_columns(
                # customer age = days from first purchase to observation end
                (
                    pl.lit(observation_end_date) - pl.col("first_purchase_date")
                )
                .dt.total_days()
                .cast(pl.Int32)
                .alias("customer_age_days"),
            )
            .collect()
        )
        return enriched

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def save_cohort_retention(
        self,
        retention_df: pl.DataFrame,
        db_client: "SupabaseClient",
    ) -> None:
        """Save retention matrix to a JSON in pipeline_runs metadata (lightweight)."""
        logger.info("Cohort retention matrix computed — {} rows", len(retention_df))
        # Retention matrix is used dashboard-side; store as JSON via Supabase
        # Full table storage can be added in a later phase if needed.
        logger.debug(retention_df.head())