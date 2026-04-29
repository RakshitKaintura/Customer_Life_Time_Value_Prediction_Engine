"""
Cold-Start LTV — Firmographic Prior.

For customers with zero transactions (fresh signups),
we use CATE estimates from the Causal ML pipeline as
a zero-shot LTV prior.

Input features (available at signup):
  - vertical         (industry)
  - company_size     (SMB / mid_market / enterprise)
  - acquisition_channel (paid_search / organic / paid_social / email)
  - plan_tier        (free / starter / professional / enterprise_trial)

Pipeline:
  1. Load CATE results from causal_model_registry
  2. Segment customers by firmographic slice
  3. Compute average CATE per slice → LTV prior
  4. Store in firmographic_ltv lookup table
  5. Serve from /score endpoint with ltv_source='firmographic_prior'
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
# Firmographic feature definitions
# ─────────────────────────────────────────────────────────────

VERTICAL_OPTIONS = [
    "healthcare", "fintech", "ecommerce", "saas",
    "manufacturing", "retail", "education", "other",
]

COMPANY_SIZE_OPTIONS = [
    "smb",        # 1–50 employees
    "mid_market", # 51–500
    "enterprise", # 500+
]

CHANNEL_OPTIONS = [
    "organic", "paid_search", "paid_social",
    "email", "referral", "direct",
]

PLAN_TIER_OPTIONS = [
    "free", "starter", "professional",
    "enterprise_trial", "enterprise",
]

# Default values when firmographic data is unknown
DEFAULT_FIRMOGRAPHIC = {
    "vertical":            "other",
    "company_size":        "smb",
    "acquisition_channel": "organic",
    "plan_tier":           "free",
}


# ─────────────────────────────────────────────────────────────
# Lookup table builder
# ─────────────────────────────────────────────────────────────

def build_firmographic_lookup(
    rfm_df: pl.DataFrame,
    cate_per_customer: dict[str, np.ndarray],
    customer_ids: list[str],
    causal_model_version: str,
) -> pl.DataFrame:
    """
    Build the firmographic LTV lookup table from CATE estimates.

    Since the UCI dataset has no firmographic data, we:
      1. Synthesise firmographic features from available proxies
      2. Compute average LTV + CATE by firmographic slice
      3. Use this as the cold-start prior

    In a real system, this would use actual CRM firmographic data.

    Returns Polars DataFrame matching firmographic_ltv table schema.
    """
    # Synthesise firmographic features from RFM proxies
    df = rfm_df.to_pandas().set_index("customer_id")

    # Map RFM features → firmographic proxies (heuristic mapping for UCI dataset)
    rng = np.random.default_rng(42)
    n = len(df)

    # country → vertical proxy
    country_to_vertical = {
        "United Kingdom": "retail",
        "Germany":        "manufacturing",
        "France":         "ecommerce",
        "EIRE":           "retail",
        "Spain":          "retail",
        "Netherlands":    "ecommerce",
        "Belgium":        "ecommerce",
        "Switzerland":    "fintech",
        "Australia":      "saas",
        "Norway":         "ecommerce",
    }

    # monetary_avg → company_size proxy
    def monetary_to_size(m: float) -> str:
        if m > 200:   return "enterprise"
        if m > 80:    return "mid_market"
        return "smb"

    # days_to_second_purchase → plan_tier proxy
    def d2s_to_tier(d: float) -> str:
        if pd.isna(d):   return "free"
        if d <= 7:        return "enterprise_trial"
        if d <= 30:       return "professional"
        if d <= 90:       return "starter"
        return "free"

    # frequency → acquisition_channel proxy
    def freq_to_channel(f: float) -> str:
        choices = CHANNEL_OPTIONS
        # Higher frequency slightly biased toward paid
        weights = [0.15, 0.30, 0.20, 0.15, 0.10, 0.10]
        return rng.choice(choices, p=weights)

    df["vertical"]            = df.get("country", pd.Series(["other"] * n, index=df.index)).map(
        lambda c: country_to_vertical.get(c, "other")
    )
    df["company_size"]        = df["monetary_avg"].apply(monetary_to_size)
    df["plan_tier"]           = df.get("days_to_second_purchase", pd.Series([365.0] * n, index=df.index)).apply(d2s_to_tier)
    df["acquisition_channel"] = [freq_to_channel(f) for f in df["frequency"]]

    # Add baseline LTV from actual_ltv_12m
    baseline_col = "actual_ltv_12m" if "actual_ltv_12m" in df.columns else "monetary_total"
    df["baseline_ltv"] = df[baseline_col].fillna(0)

    # Add CATE sum per customer (total causal uplift)
    if cate_per_customer and customer_ids:
        cate_matrix = np.column_stack(list(cate_per_customer.values()))  # (n, k)
        cate_total  = cate_matrix.sum(axis=1).clip(min=0)
        cate_series = pd.Series(cate_total, index=customer_ids)
        df["total_cate"] = df.index.map(cate_series).fillna(0)
    else:
        df["total_cate"] = 0.0

    df["ltv_prior"] = df["baseline_ltv"] + df["total_cate"] * 0.5  # 50% cate realisation

    # Aggregate by firmographic slice
    group_cols = ["vertical", "company_size", "acquisition_channel", "plan_tier"]
    agg = (
        df.groupby(group_cols)["ltv_prior"]
        .agg(
            ltv_36m_estimate="mean",
            ci_lower=lambda x: x.quantile(0.10),
            ci_upper=lambda x: x.quantile(0.90),
            n_customers="count",
        )
        .reset_index()
    )

    # Add cate_effect (mean total_cate per slice)
    cate_agg = df.groupby(group_cols)["total_cate"].mean().reset_index()
    cate_agg.columns = list(group_cols) + ["cate_effect"]
    agg = agg.merge(cate_agg, on=group_cols, how="left")

    result = pl.from_pandas(agg).with_columns([
        pl.lit(datetime.now(timezone.utc).isoformat()).alias("computed_at"),
        pl.lit(causal_model_version).alias("causal_model_version"),
        pl.col("n_customers").cast(pl.Int32),
        # Rename acquisition_channel → channel to match DB schema
    ]).rename({"acquisition_channel": "channel"})

    logger.info(
        "Firmographic lookup table: {} slices built from {} customers",
        len(result), len(df),
    )
    return result


class ColdStartScorer:
    """
    Cold-start LTV scorer for zero-transaction customers.

    Looks up the firmographic_ltv table and returns
    an LTV estimate with wide confidence interval.

    Usage:
        scorer = ColdStartScorer(db_client)
        result = scorer.score(
            vertical='healthcare',
            company_size='mid_market',
            channel='paid_search',
            plan_tier='enterprise_trial'
        )
    """

    def __init__(self, db_client: Any) -> None:
        self._db    = db_client
        self._table: pl.DataFrame | None = None

    def load_table(self) -> "ColdStartScorer":
        """Load the firmographic_ltv table from Supabase into memory."""
        rows = self._db.execute_sql(
            """
            SELECT vertical, company_size, channel, plan_tier,
                   ltv_36m_estimate, ci_lower, ci_upper, cate_effect,
                   n_customers, causal_model_version
            FROM firmographic_ltv
            ORDER BY ltv_36m_estimate DESC
            """
        )
        if rows:
            self._table = pl.DataFrame(rows)
            logger.info("Loaded {} firmographic LTV slices", len(self._table))
        else:
            logger.warning("firmographic_ltv table is empty — cold-start unavailable")
            self._table = pl.DataFrame()
        return self

    def score(
        self,
        vertical: str,
        company_size: str,
        channel: str,
        plan_tier: str,
    ) -> dict:
        """
        Score a zero-transaction customer using firmographic prior.

        Falls back to progressively coarser slices if exact match not found:
          1. Exact match (vertical, company_size, channel, plan_tier)
          2. Drop plan_tier
          3. Drop channel + plan_tier
          4. Drop company_size + channel + plan_tier
          5. Global average

        Returns:
            {ltv_36m, ci_lower, ci_upper, segment, ltv_source, match_quality, ...}
        """
        if self._table is None or len(self._table) == 0:
            return self._fallback_response(vertical, company_size, channel, plan_tier)

        # Normalise inputs
        vertical     = vertical.lower().strip()
        company_size = company_size.lower().strip()
        channel      = channel.lower().strip()
        plan_tier    = plan_tier.lower().strip()

        fallback_levels = [
            {"vertical": vertical, "company_size": company_size, "channel": channel, "plan_tier": plan_tier},
            {"vertical": vertical, "company_size": company_size, "channel": channel},
            {"vertical": vertical, "company_size": company_size},
            {"vertical": vertical},
        ]

        for level_idx, filters in enumerate(fallback_levels):
            result = self._table
            for col, val in filters.items():
                if col in result.columns:
                    result = result.filter(pl.col(col) == val)
            if len(result) > 0:
                row = result.sort("ltv_36m_estimate", descending=True)[0]
                ltv_36m  = float(row["ltv_36m_estimate"][0])
                ci_lower = float(row["ci_lower"][0])
                ci_upper = float(row["ci_upper"][0])

                match_quality = ["exact", "partial_channel", "partial_size", "vertical_only"][level_idx]

                return {
                    "ltv_source":          "firmographic_prior",
                    "ltv_36m":             ltv_36m,
                    "ltv_12m":             ltv_36m * 0.35,  # rough 12m proxy
                    "ci_lower_36m":        ci_lower,
                    "ci_upper_36m":        ci_upper,
                    "segment":             self._segment(ltv_36m),
                    "recommended_max_cac": ltv_36m * 0.40,
                    "match_quality":       match_quality,
                    "n_customers_in_slice": int(row["n_customers"][0]) if "n_customers" in row.columns else None,
                    "firmographic_inputs": {
                        "vertical":    vertical,
                        "company_size": company_size,
                        "channel":     channel,
                        "plan_tier":   plan_tier,
                    },
                }

        return self._fallback_response(vertical, company_size, channel, plan_tier)

    def _segment(self, ltv: float) -> str:
        if ltv > 10_000: return "champions"
        if ltv > 5_000:  return "high_value"
        if ltv > 1_000:  return "medium_value"
        return "low_value"

    def _fallback_response(
        self, vertical: str, company_size: str, channel: str, plan_tier: str
    ) -> dict:
        """Global average fallback when no firmographic match exists."""
        if self._table is not None and len(self._table) > 0:
            global_avg = float(self._table["ltv_36m_estimate"].mean())
            global_lo  = float(self._table["ltv_36m_estimate"].quantile(0.10))
            global_hi  = float(self._table["ltv_36m_estimate"].quantile(0.90))
        else:
            global_avg, global_lo, global_hi = 500.0, 100.0, 2000.0

        return {
            "ltv_source":          "firmographic_prior",
            "ltv_36m":             global_avg,
            "ltv_12m":             global_avg * 0.35,
            "ci_lower_36m":        global_lo,
            "ci_upper_36m":        global_hi,
            "segment":             self._segment(global_avg),
            "recommended_max_cac": global_avg * 0.40,
            "match_quality":       "global_average",
            "firmographic_inputs": {
                "vertical":    vertical,
                "company_size": company_size,
                "channel":     channel,
                "plan_tier":   plan_tier,
            },
        }

    def save_table(
        self,
        lookup_df: pl.DataFrame,
        db_client: Any,
    ) -> int:
        """Persist the firmographic lookup table to Supabase."""
        records = lookup_df.to_dicts()
        n = db_client.bulk_upsert(
            "firmographic_ltv",
            records,
            conflict_columns=["vertical", "company_size", "channel", "plan_tier"],
        )
        logger.info("Saved {} firmographic LTV rows", n)
        return n