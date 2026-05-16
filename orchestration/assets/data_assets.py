"""
Data Engineering Assets — Week 8 Dagster integration.

Asset graph:
    raw_transactions
        → cleaned_transactions
        → rfm_features
        → purchase_sequences
        → calibration_holdout_split
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from numbers import Real

import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetIn,
    Output,
    asset,
    MetadataValue,
)

from backend.features.rfm import (
    RFMPipeline,
    assign_amount_buckets,
    assign_product_categories,
    clean_transactions,
    make_calibration_holdout_split,
)
from backend.features.sequences import SequenceBuilder
from backend.data.load_data import load_uci_csv
from backend.config import settings


# ─────────────────────────────────────────────────────────────
# Raw transactions
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="data_engineering",
    description="Load and validate raw UCI Online Retail transactions",
    compute_kind="polars",
)
def raw_transactions(context) -> Output[pl.DataFrame]:
    """Load raw UCI Online Retail dataset from CSV."""
    csv_path = settings.UCI_CSV_PATH
    context.log.info(f"Loading raw transactions from {csv_path}")

    df = load_uci_csv(csv_path)

    context.log.info(f"Loaded {len(df):,} raw rows")
    return Output(
        df,
        metadata={
            "n_rows":       MetadataValue.int(len(df)),
            "n_columns":    MetadataValue.int(len(df.columns)),
            "date_range":   MetadataValue.text(
                f"{df['invoice_date'].min()} → {df['invoice_date'].max()}"
            ),
            "csv_path":     MetadataValue.path(str(csv_path)),
        },
    )


# ─────────────────────────────────────────────────────────────
# Cleaned transactions
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="data_engineering",
    description="Clean transactions — remove returns, nulls, zero-price rows",
    compute_kind="polars",
    ins={"raw_transactions": AssetIn()},
)
def cleaned_transactions(
    context,
    raw_transactions: pl.DataFrame,
) -> Output[pl.DataFrame]:
    """Apply cleaning rules and assign product categories + amount buckets."""
    context.log.info(f"Cleaning {len(raw_transactions):,} raw rows")

    cleaned = clean_transactions(raw_transactions)
    cleaned = assign_product_categories(cleaned)
    cleaned = assign_amount_buckets(cleaned)

    n_removed = len(raw_transactions) - len(cleaned)
    pct_kept  = 100 * len(cleaned) / max(len(raw_transactions), 1)

    context.log.info(f"Cleaned: {len(cleaned):,} rows ({pct_kept:.1f}% kept)")
    return Output(
        cleaned,
        metadata={
            "n_rows_cleaned":   MetadataValue.int(len(cleaned)),
            "n_rows_removed":   MetadataValue.int(n_removed),
            "pct_kept":         MetadataValue.float(float(round(pct_kept, 2))),
            "unique_customers": MetadataValue.int(cleaned["customer_id"].n_unique()),
            "unique_products":  MetadataValue.int(cleaned["stock_code"].n_unique()),
        },
    )


# ─────────────────────────────────────────────────────────────
# Calibration / holdout split
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="data_engineering",
    description="Split into calibration (6m) and holdout (6m) windows",
    compute_kind="polars",
    ins={"cleaned_transactions": AssetIn()},
)
def calibration_holdout_split(
    context,
    cleaned_transactions: pl.DataFrame,
) -> Output[dict]:
    """Create calibration/holdout temporal split."""
    context.log.info("Splitting data into calibration and holdout windows")

    calibration, holdout, obs_end, holdout_end = make_calibration_holdout_split(
        cleaned_transactions,
        observation_months=settings.OBSERVATION_WINDOW_MONTHS,
        holdout_months=settings.HOLDOUT_WINDOW_MONTHS,
    )

    result = {
        "calibration":   calibration,
        "holdout":       holdout,
        "obs_end":       obs_end,
        "holdout_end":   holdout_end,
    }

    context.log.info(
        f"Calibration: {len(calibration):,} rows (≤{obs_end}), "
        f"Holdout: {len(holdout):,} rows ({obs_end}–{holdout_end})"
    )
    return Output(
        result,
        metadata={
            "calibration_rows":  MetadataValue.int(len(calibration)),
            "holdout_rows":      MetadataValue.int(len(holdout)),
            "obs_end":           MetadataValue.text(str(obs_end)),
            "holdout_end":       MetadataValue.text(str(holdout_end)),
        },
    )


# ─────────────────────────────────────────────────────────────
# RFM features
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="data_engineering",
    description="Compute RFM features with LTV labels from holdout",
    compute_kind="polars",
    ins={"calibration_holdout_split": AssetIn()},
)
def rfm_features(
    context,
    calibration_holdout_split: dict,
) -> Output[pl.DataFrame]:
    """Build full RFM feature set with 12m holdout LTV labels."""
    split = calibration_holdout_split
    calibration: pl.DataFrame = split["calibration"]
    holdout:     pl.DataFrame = split["holdout"]
    obs_end:     date         = split["obs_end"]

    context.log.info(f"Computing RFM for {calibration['customer_id'].n_unique():,} customers")

    pipeline = RFMPipeline(calibration, observation_end_date=obs_end)
    rfm_df   = pipeline.compute()
    rfm_df   = pipeline.compute_ltv_labels(holdout, rfm_df, horizon_months=12)

    repeat_buyers = int((rfm_df["frequency"] > 0).sum())
    nonzero_ltv   = int((rfm_df["actual_ltv_12m"] > 0).sum())

    context.log.info(
        f"RFM: {len(rfm_df):,} customers, "
        f"{repeat_buyers:,} repeat buyers, "
        f"{nonzero_ltv:,} with >0 LTV"
    )

    # Persist to Supabase
    try:
        from backend.db.supabase_client import SupabaseClient
        db = SupabaseClient(use_service_role=True)
        n_saved = pipeline.save(rfm_df, db)
        context.log.info(f"Saved {n_saved:,} RFM rows to Supabase")
    except Exception as exc:
        context.log.warning(f"DB save skipped: {exc}")

    mean_frequency = rfm_df["frequency"].mean()
    mean_frequency_value = float(mean_frequency) if isinstance(mean_frequency, Real) else 0.0

    mean_ltv_12m = rfm_df.filter(pl.col("actual_ltv_12m") > 0)["actual_ltv_12m"].mean()
    mean_ltv_12m_value = float(mean_ltv_12m) if isinstance(mean_ltv_12m, Real) else 0.0

    return Output(
        rfm_df,
        metadata={
            "n_customers":    MetadataValue.int(len(rfm_df)),
            "repeat_buyers":  MetadataValue.int(repeat_buyers),
            "nonzero_ltv":    MetadataValue.int(nonzero_ltv),
            "mean_frequency": MetadataValue.float(mean_frequency_value),
            "mean_ltv_12m":   MetadataValue.float(mean_ltv_12m_value),
        },
    )


# ─────────────────────────────────────────────────────────────
# Purchase sequences
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="data_engineering",
    description="Build tokenised purchase sequences for Transformer",
    compute_kind="polars",
    ins={"calibration_holdout_split": AssetIn()},
)
def purchase_sequences(
    context,
    calibration_holdout_split: dict,
) -> Output[pl.DataFrame]:
    """Tokenise purchase sequences for Transformer training."""
    split = calibration_holdout_split
    calibration: pl.DataFrame = split["calibration"]
    obs_end:     date         = split["obs_end"]

    context.log.info("Building purchase sequences")

    builder  = SequenceBuilder(
        calibration,
        max_length            = settings.MAX_SEQUENCE_LENGTH,
        observation_end_date  = str(obs_end),
    )
    seq_df = builder.build()

    avg_len_raw = seq_df["sequence_length"].mean()
    avg_len = float(avg_len_raw) if isinstance(avg_len_raw, Real) else 0.0
    context.log.info(
        f"Built {len(seq_df):,} sequences, avg length {avg_len:.1f}"
    )

    # Persist to Supabase
    try:
        from backend.db.supabase_client import SupabaseClient
        db = SupabaseClient(use_service_role=True)
        n_saved = builder.save(seq_df, db)
        context.log.info(f"Saved {n_saved:,} sequences to Supabase")
    except Exception as exc:
        context.log.warning(f"Sequence DB save skipped: {exc}")

    max_len_raw = seq_df["sequence_length"].max()
    max_len = int(max_len_raw) if isinstance(max_len_raw, (int, float, Decimal)) else 0

    return Output(
        seq_df,
        metadata={
            "n_sequences":   MetadataValue.int(len(seq_df)),
            "avg_length":    MetadataValue.float(float(round(avg_len, 2))),
            "max_length":    MetadataValue.int(max_len),
        },
    )