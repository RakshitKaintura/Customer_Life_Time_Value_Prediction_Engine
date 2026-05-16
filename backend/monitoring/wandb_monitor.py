"""
W&B Model Registry and Performance Monitoring.

Tracks:
  - Model versions in W&B model registry
  - Validation metrics over time
  - A/B test results between model versions
  - Drift alerts as W&B alerts
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger


class WandbMonitor:
    """
    W&B monitoring client for LTV model tracking.

    Usage:
        monitor = WandbMonitor(project='ltv-prediction')
        monitor.log_model_version(model_version, metrics)
        monitor.alert_drift(psi_score=0.20)
    """

    def __init__(
        self,
        project: str = "ltv-prediction",
        entity:  str | None = None,
    ) -> None:
        self.project = project
        self.entity  = entity
        self._available = self._check_wandb()

    def _check_wandb(self) -> bool:
        try:
            import wandb  # noqa: F401
            return True
        except ImportError:
            logger.warning("wandb not installed")
            return False

    def log_model_performance(
        self,
        model_version: str,
        metrics:       dict[str, float],
        evaluation_type: str = "rolling_validation",
    ) -> None:
        """Log model performance metrics to W&B."""
        if not self._available:
            return

        try:
            import wandb
            with wandb.init(
                project = self.project,
                entity  = self.entity,
                name    = f"eval_{model_version}_{evaluation_type}",
                tags    = ["evaluation", evaluation_type, "week8"],
                config  = {"model_version": model_version},
                reinit  = True,
            ) as run:
                wandb.log({
                    **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    "evaluation_type": evaluation_type,
                    "timestamp":       datetime.now(timezone.utc).isoformat(),
                })
            logger.info("W&B performance metrics logged for {}", model_version)
        except Exception as exc:
            logger.warning("W&B logging failed: {}", exc)

    def alert_drift(
        self,
        psi_score:    float,
        alert_type:   str = "distribution_shift",
        model_version: str = "",
    ) -> None:
        """Send W&B alert when drift is detected."""
        if not self._available:
            return

        try:
            import wandb
            wandb.alert(
                title  = f"LTV Drift Alert — {alert_type}",
                text   = (
                    f"PSI score {psi_score:.4f} exceeds threshold 0.15 "
                    f"for model {model_version}. "
                    "Consider retraining."
                ),
                level  = wandb.AlertLevel.WARN,
                wait_duration = 3600,   # Don't alert again for 1 hour
            )
            logger.info("W&B drift alert sent: PSI={:.4f}", psi_score)
        except Exception as exc:
            logger.warning("W&B alert failed: {}", exc)

    def register_model(
        self,
        model_version:  str,
        model_path:     str,
        metrics:        dict[str, float],
        artifact_type:  str = "ltv_model",
    ) -> None:
        """Register a model version in the W&B model registry."""
        if not self._available:
            return

        try:
            import wandb
            with wandb.init(
                project = self.project,
                entity  = self.entity,
                name    = f"register_{model_version}",
                reinit  = True,
            ) as run:
                artifact = wandb.Artifact(
                    name     = model_version,
                    type     = artifact_type,
                    metadata = metrics,
                )
                artifact.add_dir(str(model_path))
                run.log_artifact(artifact)
            logger.info("W&B model artifact registered: {}", model_version)
        except Exception as exc:
            logger.warning("W&B model registration failed: {}", exc)