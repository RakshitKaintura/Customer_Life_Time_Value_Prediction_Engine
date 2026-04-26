"""
Optuna hyperparameter tuning for LTVTransformer.

Searches over:
  - n_layers:       2, 4, 6
  - n_heads:        4, 8
  - ffn_dim:        128, 256, 512
  - dropout:        0.05–0.30
  - learning_rate:  1e-4 – 1e-2
  - batch_size:     32, 64, 128
  - weight_decay:   1e-5 – 1e-2

Objective: minimise validation Huber loss.
Uses median pruning to terminate unpromising trials early.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

from backend.ml.sequence_dataset import PurchaseSequenceDataset, collate_fn, make_dataloaders
from backend.ml.transformer_model import LTVTransformer, build_model
from backend.ml.trainer import LTVTrainer


def objective(
    trial: Any,
    train_dataset: PurchaseSequenceDataset,
    val_dataset: PurchaseSequenceDataset,
    device: torch.device,
    quick_epochs: int = 15,
) -> float:
    """
    Optuna objective function.
    Returns validation loss (lower is better).
    """
    import optuna

    config = {
        "model_dim":     64,   # fixed for pgvector dim compatibility
        "n_layers":      trial.suggest_categorical("n_layers",  [2, 4, 6]),
        "n_heads":       trial.suggest_categorical("n_heads",   [4, 8]),
        "ffn_dim":       trial.suggest_categorical("ffn_dim",   [128, 256, 512]),
        "dropout":       trial.suggest_float("dropout",        0.05, 0.30),
        "learning_rate": trial.suggest_float("learning_rate",  1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",   1e-5, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128]),
        "grad_clip":     1.0,
        "huber_delta":   1.0,
        "max_seq_len":   50,
    }

    model = build_model(config)
    train_loader, val_loader = make_dataloaders(
        train_dataset, val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
    )

    trainer = LTVTrainer(model, config, device=device)

    try:
        history = trainer.train(
            train_loader, val_loader,
            epochs=quick_epochs,
            patience=5,
        )
    except Exception as exc:
        logger.warning("Trial {} failed: {}", trial.number, exc)
        raise optuna.exceptions.TrialPruned()

    best_val = min(history["val_loss"]) if history["val_loss"] else float("inf")
    return best_val


def run_optuna_study(
    train_dataset: PurchaseSequenceDataset,
    val_dataset: PurchaseSequenceDataset,
    n_trials: int = 20,
    study_name: str = "ltv_transformer_tuning",
    device: torch.device | None = None,
    quick_epochs: int = 15,
    db_url: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """
    Run Optuna study and return best params.

    Args:
        train_dataset:   Training PurchaseSequenceDataset
        val_dataset:     Validation PurchaseSequenceDataset
        n_trials:        Number of trials to run
        study_name:      Optuna study name
        device:          torch.device
        quick_epochs:    Epochs per trial (keep low for speed)
        db_url:          Optional SQLite URL for persistence e.g. 'sqlite:///optuna.db'

    Returns:
        (best_params dict, study object)
    """
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError:
        raise ImportError("optuna is required. Run: pip install optuna")

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = db_url or f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=storage,
        load_if_exists=True,
    )

    logger.info(
        "Starting Optuna study '{}' — {} trials on {}",
        study_name, n_trials, device,
    )

    study.optimize(
        lambda trial: objective(
            trial, train_dataset, val_dataset, device, quick_epochs
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_val    = study.best_value

    logger.info("=== Optuna Best Result ===")
    logger.info("  Best val loss: {:.6f}", best_val)
    logger.info("  Best params:   {}", best_params)

    return best_params, study