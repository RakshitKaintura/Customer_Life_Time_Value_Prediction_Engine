"""
W&B Experiment Tracking for BG/NBD models.

Provides a clean interface for logging:
  - Model parameters
  - Validation metrics
  - Calibration plots
  - Probability-alive heatmaps
  - LTV distribution charts
"""

from __future__ import annotations

import contextlib
from typing import Any, TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from backend.ml.bgnbd_model import BGNBDModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed — W&B logging disabled")


class WandbTracker:
    """
    W&B tracker for BG/NBD experiments.

    Usage:
        tracker = WandbTracker(project="ltv-prediction", name="week2_bgnbd")
        with tracker:
            tracker.log_params(model)
            tracker.log_metrics(metrics)
            tracker.log_calibration_plot(cal_data)
    """

    def __init__(
        self,
        project: str = "ltv-prediction",
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict | None = None,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.name = name
        self.tags = tags or ["week2", "bgnbd", "gamma_gamma"]
        self.config = config or {}
        self.enabled = enabled and WANDB_AVAILABLE
        self._run = None

    def __enter__(self) -> "WandbTracker":
        if self.enabled:
            self._run = wandb.init(
                project=self.project,
                name=self.name,
                tags=self.tags,
                config=self.config,
                reinit=True,
            )
        return self

    def __exit__(self, *_: Any) -> None:
        if self.enabled and self._run is not None:
            wandb.finish()

    def log_params(self, model: "BGNBDModel") -> None:
        """Log fitted model parameters to W&B config."""
        if not self.enabled:
            return
        params = model.get_params()
        wandb.config.update({
            "bgnbd_r":              params["bgnbd"]["r"],
            "bgnbd_alpha":          params["bgnbd"]["alpha"],
            "bgnbd_a":              params["bgnbd"]["a"],
            "bgnbd_b":              params["bgnbd"]["b"],
            "gg_p":                 params["gamma_gamma"]["p"],
            "gg_q":                 params["gamma_gamma"]["q"],
            "gg_v":                 params["gamma_gamma"]["v"],
            "penalizer_coef":       model.penalizer_coef,
            "model_version":        model.model_version,
            "observation_end":      str(model.observation_end),
        })

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log validation metrics to W&B."""
        if not self.enabled:
            return
        # Filter to numeric values only
        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not np.isnan(float(v))
        }
        wandb.log(numeric_metrics)
        logger.debug("Logged {} metrics to W&B", len(numeric_metrics))

    def log_calibration_plot(
        self,
        cal_data: pl.DataFrame,
        title: str = "BG/NBD Calibration: Predicted vs Actual Purchases",
    ) -> None:
        """Log calibration scatter/bar plot to W&B."""
        if not self.enabled or len(cal_data) == 0:
            return
        try:
            import wandb
            table = wandb.Table(dataframe=cal_data.to_pandas())
            wandb.log({
                "calibration_plot": wandb.plot.scatter(
                    table,
                    "predicted_purchases_avg",
                    "actual_purchases_avg",
                    title=title,
                )
            })
        except Exception as exc:
            logger.warning("W&B calibration plot failed: {}", exc)

    def log_ltv_distribution(
        self,
        predictions: pl.DataFrame,
        column: str = "ltv_36m",
        title: str = "LTV Distribution (36m)",
    ) -> None:
        """Log LTV distribution histogram to W&B."""
        if not self.enabled:
            return
        try:
            values = predictions[column].drop_nulls().to_list()
            wandb.log({
                f"{column}_distribution": wandb.Histogram(values, num_bins=50),
                f"mean_{column}":         float(np.mean(values)),
                f"median_{column}":       float(np.median(values)),
                f"p90_{column}":          float(np.percentile(values, 90)),
                f"p99_{column}":          float(np.percentile(values, 99)),
            })
        except Exception as exc:
            logger.warning("W&B LTV distribution logging failed: {}", exc)

    def log_probability_alive_matrix(
        self,
        matrix: pl.DataFrame,
    ) -> None:
        """Log P(alive) heatmap to W&B."""
        if not self.enabled:
            return
        try:
            table = wandb.Table(dataframe=matrix.to_pandas())
            wandb.log({"probability_alive_matrix": table})
        except Exception as exc:
            logger.warning("W&B P(alive) matrix logging failed: {}", exc)

    def log_predictions_table(
        self,
        predictions: pl.DataFrame,
        max_rows: int = 1000,
    ) -> None:
        """Log a sample of predictions to W&B tables."""
        if not self.enabled:
            return
        try:
            sample = predictions.sample(min(max_rows, len(predictions)))
            table = wandb.Table(dataframe=sample.to_pandas())
            wandb.log({"predictions_sample": table})
        except Exception as exc:
            logger.warning("W&B predictions table logging failed: {}", exc)

    def log_grid_search_results(
        self,
        results_df: pl.DataFrame,
    ) -> None:
        """Log grid search results to W&B."""
        if not self.enabled:
            return
        try:
            table = wandb.Table(dataframe=results_df.to_pandas())
            wandb.log({"grid_search_results": table})
        except Exception as exc:
            logger.warning("W&B grid search logging failed: {}", exc)

    def alert_metric_target(
        self,
        metrics: dict[str, float],
    ) -> None:
        """Send W&B alert if key metrics miss targets."""
        if not self.enabled:
            return
        issues = []
        if metrics.get("r2_frequency", 0) < 0.85:
            issues.append(f"BG/NBD R² = {metrics['r2_frequency']:.3f} < 0.85 target")
        if metrics.get("gini_coefficient", 0) < 0.65:
            issues.append(f"Gini = {metrics['gini_coefficient']:.3f} < 0.65 target")
        if metrics.get("top_decile_lift", 0) < 3.0:
            issues.append(f"Top decile lift = {metrics['top_decile_lift']:.2f}× < 3.0× target")
        if metrics.get("calibration_error", 1.0) > 0.10:
            issues.append(f"Calibration error = {metrics['calibration_error']:.3f} > 0.10 target")

        if issues and self._run:
            try:
                wandb.alert(
                    title="BG/NBD Metric Targets Missed",
                    text="\n".join(issues),
                    level=wandb.AlertLevel.WARN,
                )
                logger.warning("W&B alert sent: {}", "; ".join(issues))
            except Exception:
                pass