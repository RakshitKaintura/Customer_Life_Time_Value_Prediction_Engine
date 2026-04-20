"""
Purchase Sequence Builder — Polars implementation.

Builds tokenised purchase sequences for the Transformer model (Phase 3).
Defined in Week 1 so the schema is ready and data can be pre-built.

Each sequence token is a dict:
  {
    "cat_id":      int,   # product_category index
    "amount_bucket": int, # 1–5 quantile bucket
    "days_delta":  int,   # days since previous purchase (0 for first)
    "channel_id":  int,   # acquisition channel index (all 1 in UCI dataset)
  }
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from backend.db.supabase_client import SupabaseClient


# ─────────────────────────────────────────────────────────────
# Vocabularies
# ─────────────────────────────────────────────────────────────

CATEGORY_VOCAB: dict[str, int] = {
    "<PAD>": 0,
    "<UNK>": 1,
    "gift_wrap":   2,
    "homewares":   3,
    "kitchenware": 4,
    "bags_cases":  5,
    "stationery":  6,
    "seasonal":    7,
    "art_craft":   8,
    "decorative":  9,
    "novelty":    10,
    "postage":    11,
    "other":      12,
}

CHANNEL_VOCAB: dict[str, int] = {
    "<PAD>": 0,
    "<UNK>": 1,
    "organic":     2,
    "paid_search": 3,
    "paid_social": 4,
    "email":       5,
    "referral":    6,
    "direct":      7,
}

PAD_TOKEN: dict[str, int] = {
    "cat_id": 0,
    "amount_bucket": 0,
    "days_delta": 0,
    "channel_id": 0,
}


class SequenceBuilder:
    """
    Builds and pads purchase sequences per customer.

    Usage:
        builder = SequenceBuilder(transactions, max_length=50)
        sequences = builder.build()
        builder.save(sequences, db_client)
    """

    def __init__(
        self,
        transactions: pl.DataFrame,
        max_length: int = 50,
        default_channel: str = "organic",
        observation_end_date: str | None = None,
    ) -> None:
        self.transactions = transactions
        self.max_length = max_length
        self.default_channel = default_channel
        self.observation_end_date = observation_end_date

    def build(self) -> pl.DataFrame:
        """
        Build one row per customer with a padded token sequence.

        Returns DataFrame with columns:
          customer_id | sequence_json | sequence_length
        """
        logger.info(
            "Building purchase sequences for {} transactions (max_length={})",
            len(self.transactions), self.max_length,
        )

        # Aggregate per invoice (daily level)
        invoice_agg = (
            self.transactions.lazy()
            .filter(pl.col("customer_id").is_not_null())
            .filter(pl.col("quantity") > 0)
            .filter(pl.col("unit_price") > 0)
            .with_columns(
                pl.col("invoice_date").cast(pl.Date).alias("purchase_date"),
                pl.col("product_category").fill_null("other"),
                pl.col("amount_bucket").fill_null(3).cast(pl.Int32),
            )
            .group_by(["customer_id", "invoice_no", "purchase_date"])
            .agg(
                # Most common category in this invoice
                pl.col("product_category").mode().first().alias("category"),
                pl.col("amount_bucket").first().alias("amount_bucket"),
            )
            .sort(["customer_id", "purchase_date"])
            .collect()
        )

        # Compute days_delta per customer
        invoice_agg = invoice_agg.with_columns(
            pl.col("purchase_date")
                .diff()
                .dt.total_days()
                .fill_null(0)
                .clip(lower_bound=0, upper_bound=365)
                .over("customer_id")
                .cast(pl.Int32)
                .alias("days_delta")
        )

        # Map categories to IDs
        invoice_agg = invoice_agg.with_columns(
            pl.col("category")
                .replace(CATEGORY_VOCAB, default=1)
                .cast(pl.Int32)
                .alias("cat_id"),
            pl.lit(CHANNEL_VOCAB.get(self.default_channel, 1))
                .cast(pl.Int32)
                .alias("channel_id"),
        )

        # Group into sequences per customer
        sequences = (
            invoice_agg.lazy()
            .group_by("customer_id")
            .agg(
                pl.struct(["cat_id", "amount_bucket", "days_delta", "channel_id"])
                    .alias("tokens"),
                pl.len().alias("raw_length"),
            )
            .collect()
        )

        # Truncate and pad
        results = []
        for row in sequences.iter_rows(named=True):
            tokens: list[dict] = row["tokens"]
            # Truncate to last max_length purchases (most recent are most relevant)
            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length:]
            seq_len = len(tokens)
            # Left-pad with PAD_TOKEN
            padded = [PAD_TOKEN] * (self.max_length - seq_len) + tokens
            results.append({
                "customer_id": row["customer_id"],
                "sequence_json": padded,
                "sequence_length": seq_len,
                "observation_end_date": self.observation_end_date,
            })

        result_df = pl.DataFrame({
            "customer_id":         [r["customer_id"] for r in results],
            "sequence_json":       [r["sequence_json"] for r in results],
            "sequence_length":     [r["sequence_length"] for r in results],
            "observation_end_date":[r["observation_end_date"] for r in results],
        })

        logger.info(
            "Built {} sequences — avg length {:.1f}, max {}",
            len(result_df),
            result_df["sequence_length"].mean(),
            result_df["sequence_length"].max(),
        )
        return result_df

    def save(
        self,
        sequences_df: pl.DataFrame,
        db_client: "SupabaseClient",
        batch_size: int = 200,
    ) -> int:
        """Persist sequences to the purchase_sequences table."""
        import json

        records = []
        for row in sequences_df.iter_rows(named=True):
            records.append({
                "customer_id":          row["customer_id"],
                "sequence_json":        json.dumps(row["sequence_json"]),
                "sequence_length":      row["sequence_length"],
                "observation_end_date": row.get("observation_end_date"),
            })

        inserted = db_client.bulk_upsert(
            table_name="purchase_sequences",
            records=records,
            conflict_columns=["customer_id"],
            batch_size=batch_size,
        )
        logger.info("Saved {} sequences", inserted)
        return inserted

    @staticmethod
    def vocab_sizes() -> dict[str, int]:
        return {
            "category_vocab_size": len(CATEGORY_VOCAB),
            "channel_vocab_size":  len(CHANNEL_VOCAB),
            "amount_bucket_max":   5,
            "max_days_delta":      365,
        }