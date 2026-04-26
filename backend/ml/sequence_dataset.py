"""
PyTorch Dataset for purchase sequences.

Reads from Polars DataFrames (pre-built by SequenceBuilder in Week 1)
and serves tokenised tensors + LTV labels to the DataLoader.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader


class PurchaseSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping purchase sequences and LTV labels.

    Each sample:
        tokens:  dict of LongTensors (cat_id, amount_bucket, days_delta, channel_id)
                 each shape (max_length,)
        targets: dict of FloatTensors (ltv_12m, ltv_24m, ltv_36m)
        customer_id: str
    """

    def __init__(
        self,
        sequences: pl.DataFrame,
        rfm_labels: pl.DataFrame,
        max_length: int = 50,
        ltv_12m_col: str = "actual_ltv_12m",
        ltv_24m_col: str | None = None,
        ltv_36m_col: str | None = None,
    ) -> None:
        """
        Args:
            sequences:   Polars DataFrame from SequenceBuilder.build()
                         Columns: customer_id, sequence_json, sequence_length
            rfm_labels:  Polars DataFrame with LTV label columns
            max_length:  Pad/truncate all sequences to this length
        """
        self.max_length = max_length

        # Join sequences with labels
        joined = sequences.join(
            rfm_labels.select(
                ["customer_id"]
                + [c for c in [ltv_12m_col, ltv_24m_col, ltv_36m_col] if c]
            ),
            on="customer_id",
            how="inner",
        )

        self.customer_ids  = joined["customer_id"].to_list()
        self.sequences_raw = joined["sequence_json"].to_list()

        # LTV labels — fill nulls with 0
        self.ltv_12m = torch.FloatTensor(
            joined[ltv_12m_col].fill_null(0).to_list()
        )
        if ltv_24m_col and ltv_24m_col in joined.columns:
            self.ltv_24m = torch.FloatTensor(joined[ltv_24m_col].fill_null(0).to_list())
        else:
            self.ltv_24m = self.ltv_12m * 1.8   # rough proxy

        if ltv_36m_col and ltv_36m_col in joined.columns:
            self.ltv_36m = torch.FloatTensor(joined[ltv_36m_col].fill_null(0).to_list())
        else:
            self.ltv_36m = self.ltv_12m * 2.5   # rough proxy

    def __len__(self) -> int:
        return len(self.customer_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self.sequences_raw[idx]

        # Parse JSON if stored as string
        if isinstance(raw, str):
            tokens_list = json.loads(raw)
        else:
            tokens_list = raw   # already a list of dicts

        # Build per-feature lists
        cat_ids      = [t.get("cat_id",       0) for t in tokens_list]
        buckets      = [t.get("amount_bucket", 0) for t in tokens_list]
        days_deltas  = [t.get("days_delta",    0) for t in tokens_list]
        channel_ids  = [t.get("channel_id",    0) for t in tokens_list]

        # Truncate if over max_length
        if len(cat_ids) > self.max_length:
            cat_ids     = cat_ids[-self.max_length:]
            buckets     = buckets[-self.max_length:]
            days_deltas = days_deltas[-self.max_length:]
            channel_ids = channel_ids[-self.max_length:]

        # Left-pad to max_length
        pad = self.max_length - len(cat_ids)
        cat_ids     = [0] * pad + cat_ids
        buckets     = [0] * pad + buckets
        days_deltas = [0] * pad + days_deltas
        channel_ids = [0] * pad + channel_ids

        return {
            "tokens": {
                "cat_id":        torch.LongTensor(cat_ids),
                "amount_bucket": torch.LongTensor(buckets),
                "days_delta":    torch.LongTensor(days_deltas),
                "channel_id":    torch.LongTensor(channel_ids),
            },
            "targets": {
                "ltv_12m": self.ltv_12m[idx],
                "ltv_24m": self.ltv_24m[idx],
                "ltv_36m": self.ltv_36m[idx],
            },
            "customer_id": self.customer_ids[idx],
        }


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """
    Custom collate for DataLoader.
    Stacks token tensors and targets; keeps customer_ids as list.
    """
    return {
        "tokens": {
            k: torch.stack([b["tokens"][k] for b in batch])
            for k in batch[0]["tokens"]
        },
        "targets": {
            k: torch.stack([b["targets"][k] for b in batch])
            for k in batch[0]["targets"]
        },
        "customer_ids": [b["customer_id"] for b in batch],
    }


def make_dataloaders(
    train_dataset: PurchaseSequenceDataset,
    val_dataset: PurchaseSequenceDataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader