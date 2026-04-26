"""
Transformer LTV Model Trainer.

Handles:
  - Training loop with gradient clipping
  - Learning rate scheduling (CosineAnnealing with warmup)
  - Early stopping
  - Checkpoint saving (best val loss)
  - W&B metric logging per epoch
  - Evaluation on validation set
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from backend.ml.transformer_model import LTVTransformer, MultiHorizonHuberLoss


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class WarmupScheduler:
    """Linear warmup then cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.current_step  = 0
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]

    def step(self) -> None:
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


class LTVTrainer:
    """
    Full training loop for LTVTransformer.

    Usage:
        trainer = LTVTrainer(model, config)
        history = trainer.train(train_loader, val_loader)
        trainer.save_checkpoint(path)
    """

    def __init__(
        self,
        model: LTVTransformer,
        config: dict[str, Any],
        device: torch.device | None = None,
        wandb_run: Any | None = None,
    ) -> None:
        self.model   = model
        self.config  = config
        self.wandb   = wandb_run
        self.device  = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps")  if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)
        logger.info("Trainer device: {}", self.device)

        self.criterion = MultiHorizonHuberLoss(
            delta=config.get("huber_delta", 1.0),
            weights=tuple(config.get("loss_weights", [0.3, 0.3, 0.4])),
            positive_weight=config.get("positive_ltv_weight", 1.0),
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "val_mae_12m": [], "val_mae_36m": [], "lr": [],
        }
        self.best_val_loss   = float("inf")
        self.best_epoch      = 0
        self.best_state_dict = None

    def _to_device(
        self, tokens: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict, dict]:
        tokens  = {k: v.to(self.device) for k, v in tokens.items()}
        targets = {k: v.to(self.device) for k, v in targets.items()}
        return tokens, targets

    def _train_epoch(self, loader: DataLoader, scheduler: Any) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            tokens, targets = self._to_device(batch["tokens"], batch["targets"])

            self.optimizer.zero_grad()
            outputs = self.model(tokens)
            loss    = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get("grad_clip", 1.0),
            )

            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_pred_12m, all_pred_36m = [], []
        all_true_12m, all_true_36m = [], []

        for batch in loader:
            tokens, targets = self._to_device(batch["tokens"], batch["targets"])
            outputs  = self.model(tokens)
            loss     = self.criterion(outputs, targets)

            total_loss += loss.item()
            n_batches  += 1

            all_pred_12m.extend(outputs["ltv_12m"].cpu().numpy())
            all_pred_36m.extend(outputs["ltv_36m"].cpu().numpy())
            all_true_12m.extend(targets["ltv_12m"].cpu().numpy())
            all_true_36m.extend(targets["ltv_36m"].cpu().numpy())

        p12 = np.array(all_pred_12m)
        p36 = np.array(all_pred_36m)
        t12 = np.array(all_true_12m)
        t36 = np.array(all_true_36m)

        mean_ltv_12m = t12.mean() if t12.mean() > 0 else 1.0

        return {
            "val_loss":    total_loss / max(n_batches, 1),
            "val_mae_12m": float(np.mean(np.abs(t12 - p12))),
            "val_mae_36m": float(np.mean(np.abs(t36 - p36))),
            "val_mae_pct": float(np.mean(np.abs(t12 - p12)) / mean_ltv_12m),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """
        Full training loop.

        Returns:
            history dict with loss curves
        """
        early_stop = EarlyStopping(patience=patience)
        total_steps = epochs * len(train_loader)
        warmup_steps = min(500, total_steps // 10)

        scheduler = WarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        logger.info(
            "Training {} epochs | {} steps | warmup={} | device={}",
            epochs, total_steps, warmup_steps, self.device,
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader, scheduler)
            val_metrics = self._eval_epoch(val_loader)
            elapsed = time.time() - t0

            current_lr = scheduler.get_last_lr()[0]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_mae_12m"].append(val_metrics["val_mae_12m"])
            self.history["val_mae_36m"].append(val_metrics["val_mae_36m"])
            self.history["lr"].append(current_lr)

            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss   = val_metrics["val_loss"]
                self.best_epoch      = epoch
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, tag="best")

            if self.wandb is not None:
                try:
                    import wandb
                    wandb.log({
                        "epoch":       epoch,
                        "train_loss":  train_loss,
                        "val_loss":    val_metrics["val_loss"],
                        "val_mae_12m": val_metrics["val_mae_12m"],
                        "val_mae_36m": val_metrics["val_mae_36m"],
                        "val_mae_pct": val_metrics["val_mae_pct"],
                        "lr":          current_lr,
                    })
                except Exception:
                    pass

            if epoch % 5 == 0 or epoch <= 5:
                logger.info(
                    "Epoch {:3d}/{} | train={:.4f} val={:.4f} mae12m={:.2f} "
                    "mae36m={:.2f} lr={:.2e} [{:.1f}s]",
                    epoch, epochs,
                    train_loss, val_metrics["val_loss"],
                    val_metrics["val_mae_12m"], val_metrics["val_mae_36m"],
                    current_lr, elapsed,
                )

            if early_stop.step(val_metrics["val_loss"]):
                logger.info(
                    "Early stopping at epoch {} (best val={:.4f} at epoch {})",
                    epoch, self.best_val_loss, self.best_epoch,
                )
                break

        # Restore best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info(
                "Restored best weights from epoch {} (val_loss={:.4f})",
                self.best_epoch, self.best_val_loss,
            )

        return self.history

    def save_checkpoint(
        self,
        directory: str | Path,
        tag: str = "latest",
    ) -> Path:
        """Save model state dict + config."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"transformer_{tag}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config":           self.config,
            "best_val_loss":    self.best_val_loss,
            "best_epoch":       self.best_epoch,
            "history":          self.history,
        }, path)
        logger.info("Checkpoint saved → {}", path)
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model: LTVTransformer,
    ) -> "LTVTrainer":
        """Load trainer state from checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        trainer = cls(model, ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.best_val_loss = ckpt["best_val_loss"]
        trainer.best_epoch    = ckpt["best_epoch"]
        trainer.history       = ckpt["history"]
        logger.info("Loaded checkpoint from {} (best_val_loss={:.4f})", path, trainer.best_val_loss)
        return trainer
