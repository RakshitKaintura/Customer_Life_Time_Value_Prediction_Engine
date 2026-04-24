"""
BG/NBD + Gamma-Gamma Hyperparameter Tuning via scipy.optimize.

Searches for the best penalizer_coef values that maximise
holdout validation metrics (MAE, Gini, decile lift).

Also provides a grid search fallback for simple sweeps.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl
from loguru import logger
from scipy.optimize import minimize_scalar

from backend.ml.bgnbd_model import BGNBDModel


# ─────────────────────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────────────────────

def _objective(
    log_penalizer: float,
    calibration_rfm: pl.DataFrame,
    holdout_rfm: pl.DataFrame,
    observation_end: Any,
) -> float:
    """
    Objective: minimise MAE (12m) on holdout.
    Uses log-space penalizer to keep optimisation numerically stable.
    """
    penalizer = float(np.exp(log_penalizer))
    penalizer = max(1e-6, min(penalizer, 10.0))  # clip to [1e-6, 10]

    try:
        model = BGNBDModel(
            penalizer_coef=penalizer,
            observation_end=observation_end,
        )
        model.fit(calibration_rfm, verbose=False)
        metrics = model.validate(calibration_rfm, holdout_rfm)

        # Primary: MAE as % of mean LTV (lower is better)
        # Secondary: penalise if Gini drops below 0.65
        loss = metrics["mae_pct_12m"]
        if metrics.get("gini_coefficient", 1.0) < 0.65:
            loss += 1.0   # penalty for poor Gini
        if metrics.get("top_decile_lift", 10.0) < 3.0:
            loss += 0.5   # penalty for poor lift

        logger.debug(
            "penalizer={:.6f}  mae_pct={:.4f}  gini={:.4f}  lift={:.2f}  loss={:.4f}",
            penalizer,
            metrics["mae_pct_12m"],
            metrics.get("gini_coefficient", 0),
            metrics.get("top_decile_lift", 0),
            loss,
        )
        return float(loss)

    except Exception as exc:
        logger.warning("Objective failed for penalizer={:.6f}: {}", penalizer, exc)
        return 999.0


# ─────────────────────────────────────────────────────────────
# Scipy optimizer
# ─────────────────────────────────────────────────────────────

def tune_penalizer_scipy(
    calibration_rfm: pl.DataFrame,
    holdout_rfm: pl.DataFrame,
    observation_end: Any,
    bounds: tuple[float, float] = (1e-6, 5.0),
    method: str = "bounded",
) -> tuple[float, dict]:
    """
    Use scipy.optimize.minimize_scalar to find the best penalizer_coef.

    Args:
        calibration_rfm: Polars RFM DataFrame (calibration window)
        holdout_rfm:     Polars RFM DataFrame (holdout window)
        observation_end: date of observation end
        bounds:          (min, max) penalizer range
        method:          scipy method ('bounded' uses Brent's method)

    Returns:
        (best_penalizer, result_dict)
    """
    logger.info(
        "Tuning penalizer_coef via scipy.minimize_scalar (bounds={})",
        bounds,
    )

    log_bounds = (np.log(bounds[0]), np.log(bounds[1]))

    result = minimize_scalar(
        fun=_objective,
        bounds=log_bounds,
        method=method,
        args=(calibration_rfm, holdout_rfm, observation_end),
        options={"xatol": 1e-3, "maxiter": 20},
    )

    best_penalizer = float(np.exp(result.x))
    logger.info(
        "Best penalizer: {:.6f}  (loss={:.4f}, nfev={})",
        best_penalizer, result.fun, result.nfev,
    )

    return best_penalizer, {
        "best_penalizer": best_penalizer,
        "best_loss":      float(result.fun),
        "n_evaluations":  int(result.nfev),
        "converged":      result.success if hasattr(result, "success") else True,
        "method":         method,
    }


# ─────────────────────────────────────────────────────────────
# Grid search (simpler, more interpretable)
# ─────────────────────────────────────────────────────────────

def tune_penalizer_grid(
    calibration_rfm: pl.DataFrame,
    holdout_rfm: pl.DataFrame,
    observation_end: Any,
    penalizer_values: list[float] | None = None,
) -> tuple[float, pl.DataFrame]:
    """
    Grid search over a list of penalizer_coef values.

    Returns:
        (best_penalizer, results_dataframe)
    """
    if penalizer_values is None:
        penalizer_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    logger.info(
        "Grid search over {} penalizer values: {}",
        len(penalizer_values), penalizer_values,
    )

    rows = []
    for penalizer in penalizer_values:
        try:
            model = BGNBDModel(
                penalizer_coef=penalizer,
                observation_end=observation_end,
            )
            model.fit(calibration_rfm, verbose=False)
            metrics = model.validate(calibration_rfm, holdout_rfm)
            rows.append({
                "penalizer":         penalizer,
                "mae_pct_12m":       metrics["mae_pct_12m"],
                "mae_ltv_12m":       metrics["mae_ltv_12m"],
                "gini_coefficient":  metrics["gini_coefficient"],
                "top_decile_lift":   metrics["top_decile_lift"],
                "calibration_error": metrics["calibration_error"],
                "r2_frequency":      metrics["r2_frequency"],
                "bgnbd_ll":          model._fit_metrics["bgnbd_log_likelihood"],
            })
            logger.info(
                "penalizer={:.4f}  mae_pct={:.4f}  gini={:.4f}  lift={:.2f}",
                penalizer,
                metrics["mae_pct_12m"],
                metrics["gini_coefficient"],
                metrics["top_decile_lift"],
            )
        except Exception as exc:
            logger.warning("penalizer={} failed: {}", penalizer, exc)
            rows.append({
                "penalizer":         penalizer,
                "mae_pct_12m":       999.0,
                "mae_ltv_12m":       None,
                "gini_coefficient":  None,
                "top_decile_lift":   None,
                "calibration_error": None,
                "r2_frequency":      None,
                "bgnbd_ll":          None,
            })

    results_df = pl.DataFrame(rows).sort("mae_pct_12m")
    best_penalizer = float(results_df["penalizer"][0])
    logger.info("Best penalizer (grid): {:.6f}", best_penalizer)
    return best_penalizer, results_df
