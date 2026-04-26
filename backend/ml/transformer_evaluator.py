"""
Holdout Evaluation for the Transformer LTV model.

Mirrors the same metrics as the BG/NBD evaluator so
results are directly comparable before fusion in Phase 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

from backend.ml.bgnbd_model import (
    compute_gini,
    compute_top_decile_lift,
    compute_calibration_error,
)
from backend.ml.sequence_dataset import PurchaseSequenceDataset
from backend.ml.transformer_model import LTVTransformer
from backend.ml.sequence_dataset import collate_fn


@torch.no_grad()
def evaluate_on_holdout(
    model: LTVTransformer,
    holdout_loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Run the trained Transformer on the holdout set and compute metrics.

    Returns:
        mae_ltv_12m, mae_ltv_36m, mae_pct_12m, gini_coefficient,
        top_decile_lift, calibration_error, rmse_ltv_12m
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    model.to(device)

    all_pred_12m, all_pred_36m = [], []
    all_true_12m, all_true_36m = [], []

    for batch in holdout_loader:
        tokens  = {k: v.to(device) for k, v in batch["tokens"].items()}
        targets = batch["targets"]

        outputs = model(tokens)

        all_pred_12m.extend(outputs["ltv_12m"].cpu().numpy())
        all_pred_36m.extend(outputs["ltv_36m"].cpu().numpy())
        all_true_12m.extend(targets["ltv_12m"].numpy())
        all_true_36m.extend(targets["ltv_36m"].numpy())

    p12 = np.array(all_pred_12m)
    p36 = np.array(all_pred_36m)
    t12 = np.array(all_true_12m)
    t36 = np.array(all_true_36m)

    mean_ltv = float(t12.mean()) if t12.mean() > 0 else 1.0

    metrics = {
        "mae_ltv_12m":       float(np.mean(np.abs(t12 - p12))),
        "mae_ltv_36m":       float(np.mean(np.abs(t36 - p36))),
        "mae_pct_12m":       float(np.mean(np.abs(t12 - p12)) / mean_ltv),
        "rmse_ltv_12m":      float(np.sqrt(np.mean((t12 - p12) ** 2))),
        "gini_coefficient":  compute_gini(t12, p12),
        "top_decile_lift":   compute_top_decile_lift(t12, p12),
        "calibration_error": compute_calibration_error(t12, p12),
        "mean_actual_ltv":   mean_ltv,
        "n_customers":       len(t12),
    }

    logger.info("=== Transformer Holdout Metrics ===")
    logger.info("  MAE LTV 12m:        £{:.2f}  ({:.1f}% of mean)",
                metrics["mae_ltv_12m"], 100 * metrics["mae_pct_12m"])
    logger.info("  MAE LTV 36m:        £{:.2f}", metrics["mae_ltv_36m"])
    logger.info("  Gini coefficient:   {:.4f}  (target > 0.65)", metrics["gini_coefficient"])
    logger.info("  Top decile lift:    {:.2f}×  (target > 3.0×)", metrics["top_decile_lift"])
    logger.info("  Calibration error:  {:.4f}  (target < 0.10)", metrics["calibration_error"])

    return metrics


@torch.no_grad()
def predict_all_customers(
    model: LTVTransformer,
    dataset: PurchaseSequenceDataset,
    batch_size: int = 128,
    device: torch.device | None = None,
    n_mc_samples: int = 50,
) -> pl.DataFrame:
    """
    Generate full predictions + MC Dropout uncertainty for all customers.

    Returns Polars DataFrame with one row per customer.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    rows = []
    for batch in loader:
        tokens = {k: v.to(device) for k, v in batch["tokens"].items()}

        # Deterministic prediction
        det_out = model(tokens)

        # Monte Carlo Dropout uncertainty
        mc_out = model.predict_with_uncertainty(tokens, n_samples=n_mc_samples)

        for i, cid in enumerate(batch["customer_ids"]):
            rows.append({
                "customer_id":      cid,
                "ltv_12m":          float(det_out["ltv_12m"][i]),
                "ltv_24m":          float(det_out["ltv_24m"][i]),
                "ltv_36m":          float(det_out["ltv_36m"][i]),
                "ltv_12m_mean":     float(mc_out["ltv_12m_mean"][i]),
                "ltv_12m_std":      float(mc_out["ltv_12m_std"][i]),
                "ltv_36m_mean":     float(mc_out["ltv_36m_mean"][i]),
                "ltv_36m_std":      float(mc_out["ltv_36m_std"][i]),
                "ltv_12m_lower":    float(mc_out["ltv_12m_lower"][i]),
                "ltv_12m_upper":    float(mc_out["ltv_12m_upper"][i]),
                "ltv_36m_lower":    float(mc_out["ltv_36m_lower"][i]),
                "ltv_36m_upper":    float(mc_out["ltv_36m_upper"][i]),
            })

    return pl.DataFrame(rows)
