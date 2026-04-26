"""Unit tests for PurchaseSequenceDataset."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import torch

from backend.ml.sequence_dataset import PurchaseSequenceDataset, collate_fn


def _make_sequences(n: int = 20, max_len: int = 10) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        length = rng.integers(2, max_len + 1)
        tokens = [
            {
                "cat_id":        int(rng.integers(1, 13)),
                "amount_bucket": int(rng.integers(1, 6)),
                "days_delta":    int(rng.integers(0, 100)),
                "channel_id":    int(rng.integers(1, 8)),
            }
            for _ in range(length)
        ]
        rows.append({
            "customer_id":    f"C{i:04d}",
            "sequence_json":  json.dumps(tokens),
            "sequence_length": length,
        })
    return pl.DataFrame(rows)


def _make_labels(n: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "customer_id":    [f"C{i:04d}" for i in range(n)],
        "actual_ltv_12m": rng.uniform(0, 1000, n).tolist(),
    })


def test_dataset_length() -> None:
    seq = _make_sequences(20)
    lbl = _make_labels(20)
    ds = PurchaseSequenceDataset(seq, lbl, max_length=10)
    assert len(ds) == 20


def test_dataset_item_keys() -> None:
    seq = _make_sequences(5)
    lbl = _make_labels(5)
    ds  = PurchaseSequenceDataset(seq, lbl, max_length=10)
    item = ds[0]
    assert "tokens"  in item
    assert "targets" in item
    assert "customer_id" in item


def test_token_shapes() -> None:
    MAX = 10
    seq = _make_sequences(5)
    lbl = _make_labels(5)
    ds  = PurchaseSequenceDataset(seq, lbl, max_length=MAX)
    item = ds[0]
    for k in ["cat_id", "amount_bucket", "days_delta", "channel_id"]:
        assert item["tokens"][k].shape == (MAX,), f"{k} shape wrong"


def test_token_dtypes() -> None:
    seq = _make_sequences(5)
    lbl = _make_labels(5)
    ds  = PurchaseSequenceDataset(seq, lbl, max_length=10)
    item = ds[0]
    for k, v in item["tokens"].items():
        assert v.dtype == torch.long, f"{k} should be long"


def test_ltv_non_negative() -> None:
    seq = _make_sequences(10)
    lbl = _make_labels(10)
    ds  = PurchaseSequenceDataset(seq, lbl, max_length=10)
    for i in range(len(ds)):
        assert ds[i]["targets"]["ltv_12m"].item() >= 0


def test_collate_fn_stacks() -> None:
    seq = _make_sequences(8)
    lbl = _make_labels(8)
    ds  = PurchaseSequenceDataset(seq, lbl, max_length=10)
    batch = collate_fn([ds[i] for i in range(4)])
    assert batch["tokens"]["cat_id"].shape == (4, 10)
    assert len(batch["customer_ids"]) == 4


def test_padding_is_zero() -> None:
    """Short sequences should be left-padded with zeros."""
    seq = pl.DataFrame([{
        "customer_id":    "C0000",
        "sequence_json":  json.dumps([{"cat_id": 3, "amount_bucket": 2, "days_delta": 5, "channel_id": 1}]),
        "sequence_length": 1,
    }])
    lbl = pl.DataFrame({"customer_id": ["C0000"], "actual_ltv_12m": [500.0]})
    ds  = PurchaseSequenceDataset(seq, lbl, max_length=5)
    item = ds[0]
    # First 4 positions should be padding (0)
    assert item["tokens"]["cat_id"][:4].tolist() == [0, 0, 0, 0]
    # Last position should be the actual token
    assert item["tokens"]["cat_id"][4].item() == 3