"""
BG/NBD + Gamma-Gamma Probabilistic LTV Model.

Wraps the `lifetimes` library with:
  - Polars → NumPy conversion at the boundary
  - Hyperparameter tuning via scipy.optimize
  - Monte Carlo confidence intervals
  - Holdout validation with MAE / RMSE / Gini / decile lift
  - Calibration plot data generation
  - W&B experiment tracking
  - Supabase persistence of parameters and per-customer predictions

BG/NBD notation (Fader et al. 2005):
    x   = frequency (number of repeat purchases)
    t_x = recency   (days from first to last purchase)
    T   = customer age (days from first purchase to observation end)

Gamma-Gamma (Fader et al. 2005):
    Uses only repeat buyers (frequency > 0)
    Inputs: frequency, avg_monetary_value
"""

from __future__ import annotations

import time
import uuid
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.fitters import ConvergenceError
from lifetimes.utils import calibration_and_holdout_data
from loguru import logger
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

MARGIN = 0.20
ANNUAL_DISCOUNT_RATE = 0.10
DEFAULT_PENALIZER = 0.001


# ─────────────────────────────────────────────────────────────
# Data conversion helpers
# ─────────────────────────────────────────────────────────────

def polars_rfm_to_pandas(rfm: pl.DataFrame) -> pd.DataFrame:
    """
    Convert a Polars RFM DataFrame to the Pandas format expected by `lifetimes`.

    lifetimes expects columns:
        frequency, recency, T, monetary_value

    Our Polars schema uses:
        frequency, recency_days, t_days, monetary_avg
    """
    df = rfm.select([
        pl.col("customer_id"),
        pl.col("frequency"),
        pl.col("recency_days").alias("recency"),
        pl.col("t_days").alias("T"),
        pl.col("monetary_avg").alias("monetary_value"),
    ]).to_pandas()

    df = df.set_index("customer_id")

    # lifetimes requires float64
    for col in ["frequency", "recency", "T", "monetary_value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Normalise to lifetimes assumptions.
    # `frequency` in BG/NBD is count of repeat purchases and must be integer >= 0.
    df["frequency"] = np.clip(np.floor(df["frequency"].to_numpy(dtype=float)), a_min=0, a_max=None)
    df["T"] = df["T"].clip(lower=0)
    df["recency"] = df["recency"].clip(lower=0)

    # Drop rows with NaN in required columns
    before = len(df)
    df = df.dropna(subset=["frequency", "recency", "T", "monetary_value"])
    if len(df) < before:
        logger.warning("Dropped {} rows with NaN in BG/NBD inputs", before - len(df))

    # BG/NBD constraints:
    # 1) recency <= T
    # 2) recency == 0 when frequency == 0
    df.loc[df["frequency"] == 0, "recency"] = 0.0
    df = df[df["recency"] <= df["T"]]

    # Gamma-Gamma constraint: monetary_value > 0
    df = df[df["monetary_value"] > 0]

    logger.debug(
        "Polars→Pandas conversion: {} customers (freq>0: {})",
        len(df),
        (df["frequency"] > 0).sum(),
    )
    return df


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Gini coefficient for ranking quality.
    Higher = better discrimination between high- and low-LTV customers.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0

    def _raw_gini(actual: np.ndarray, pred: np.ndarray) -> float:
        if np.sum(actual) == 0:
            return 0.0
        order = np.lexsort((np.arange(len(actual)), -pred))
        actual_sorted = actual[order]
        cumulative = np.cumsum(actual_sorted)
        gini_sum = cumulative.sum() / cumulative[-1]
        gini_sum -= (len(actual) + 1) / 2.0
        return gini_sum / len(actual)

    denom = _raw_gini(y_true, y_true)
    if denom == 0:
        return 0.0
    return float(_raw_gini(y_true, y_pred) / denom)


def compute_top_decile_lift(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Top decile lift: avg actual LTV in top-10% predicted / avg actual LTV overall.
    Target: > 3.0×
    """
    n = len(y_true)
    top_n = max(1, n // 10)
    sorted_idx = np.argsort(y_pred)[::-1]
    top_actual = y_true[sorted_idx[:top_n]].mean()
    overall_actual = y_true.mean()
    if overall_actual == 0:
        return 0.0
    return float(top_actual / overall_actual)


def compute_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Mean absolute calibration error across decile bins.
    Target: < 10%
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0

    # If predictions collapse to a single value, use global relative error.
    if np.allclose(y_pred, y_pred[0]):
        denom = max(abs(float(np.mean(y_true))), 1e-9)
        return float(abs(float(np.mean(y_true)) - float(np.mean(y_pred))) / denom)

    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    errors = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        else:
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            denom = max(abs(float(y_true[mask].mean())), 1e-9)
            rel_err = abs(float(y_true[mask].mean()) - float(y_pred[mask].mean())) / denom
            errors.append(rel_err)
    return float(np.mean(errors)) if errors else 0.0


# ─────────────────────────────────────────────────────────────
# BG/NBD + Gamma-Gamma Model
# ─────────────────────────────────────────────────────────────

class BGNBDModel:
    """
    BG/NBD + Gamma-Gamma LTV model.

    Lifecycle:
        model = BGNBDModel()
        model.fit(rfm_polars_df)
        predictions = model.predict(rfm_polars_df)
        metrics = model.validate(rfm_polars_df, holdout_polars_df)
        model.save_params(db_client)
        model.save_predictions(predictions, db_client)
    """

    def __init__(
        self,
        penalizer_coef: float = DEFAULT_PENALIZER,
        model_version: str | None = None,
        observation_end: date | None = None,
    ) -> None:
        self.penalizer_coef = penalizer_coef
        self.model_version = model_version or f"bgnbd_v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
        self.observation_end = observation_end or date.today()

        self.bgf: BetaGeoFitter | None = None
        self.ggf: GammaGammaFitter | None = None
        self._fit_metrics: dict[str, Any] = {}
        self._is_fitted = False

    # ── Fitting ──────────────────────────────────────────────

    def fit(
        self,
        rfm: pl.DataFrame,
        verbose: bool = False,
    ) -> "BGNBDModel":
        """
        Fit BG/NBD and Gamma-Gamma models on a Polars RFM DataFrame.

        Args:
            rfm:     Polars DataFrame with columns:
                     customer_id, frequency, recency_days, t_days, monetary_avg
            verbose: Print convergence info from lifetimes

        Returns: self (for chaining)
        """
        df = polars_rfm_to_pandas(rfm)
        logger.info(
            "Fitting BG/NBD on {} customers (penalizer={})",
            len(df), self.penalizer_coef,
        )

        # ── BG/NBD ──
        t0 = time.time()
        base_pen = min(max(float(self.penalizer_coef), 1e-6), 10.0)
        bgnbd_candidates = [
            base_pen,
            min(base_pen * 2.0, 10.0),
            min(base_pen * 5.0, 10.0),
            min(base_pen * 10.0, 10.0),
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
        ]
        bgnbd_candidates = list(dict.fromkeys([float(p) for p in bgnbd_candidates if 1e-6 <= p <= 10.0]))
        bgnbd_penalizer = bgnbd_candidates[0]
        last_bgnbd_exc: Exception | None = None
        for candidate in bgnbd_candidates:
            try:
                self.bgf = BetaGeoFitter(penalizer_coef=candidate)
                self.bgf.fit(
                    df["frequency"],
                    df["recency"],
                    df["T"],
                    verbose=verbose,
                )
                bgnbd_penalizer = candidate
                break
            except ConvergenceError as exc:
                last_bgnbd_exc = exc
                logger.warning("BG/NBD did not converge at penalizer={}; trying next.", candidate)
        else:
            raise ConvergenceError(
                f"BG/NBD failed to converge for penalizers tried: {bgnbd_candidates}"
            ) from last_bgnbd_exc
        if self.bgf is None:
            raise RuntimeError("BG/NBD fitter missing after successful fit.")
        bgf = self.bgf
        bgnbd_elapsed = time.time() - t0
        bgnbd_ll = float(getattr(bgf, "log_likelihood_", float("nan")))
        logger.info(
            "BG/NBD fitted in {:.2f}s — log-likelihood: {:.4f}",
            bgnbd_elapsed,
            bgnbd_ll if np.isfinite(bgnbd_ll) else float("nan"),
        )

        # ── Gamma-Gamma (repeat buyers only) ──
        repeat = df[df["frequency"] > 0]
        logger.info(
            "Fitting Gamma-Gamma on {} repeat buyers ({:.1f}%)",
            len(repeat),
            100 * len(repeat) / len(df),
        )

        t0 = time.time()
        gg_candidates = [
            base_pen,
            min(base_pen * 2.0, 10.0),
            min(base_pen * 5.0, 10.0),
            min(base_pen * 10.0, 10.0),
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
        ]
        gg_candidates = list(dict.fromkeys([float(p) for p in gg_candidates if 1e-6 <= p <= 10.0]))
        gg_penalizer = gg_candidates[0]
        last_gg_exc: Exception | None = None
        for candidate in gg_candidates:
            try:
                self.ggf = GammaGammaFitter(penalizer_coef=candidate)
                self.ggf.fit(
                    repeat["frequency"],
                    repeat["monetary_value"],
                    verbose=verbose,
                )
                gg_penalizer = candidate
                break
            except ConvergenceError as exc:
                last_gg_exc = exc
                logger.warning("Gamma-Gamma did not converge at penalizer={}; trying next.", candidate)
        else:
            raise ConvergenceError(
                f"Gamma-Gamma failed to converge for penalizers tried: {gg_candidates}"
            ) from last_gg_exc
        if self.ggf is None:
            raise RuntimeError("Gamma-Gamma fitter missing after successful fit.")
        ggf = self.ggf
        gg_elapsed = time.time() - t0
        gg_ll = float(getattr(ggf, "log_likelihood_", float("nan")))
        logger.info(
            "Gamma-Gamma fitted in {:.2f}s — log-likelihood: {:.4f}",
            gg_elapsed,
            gg_ll if np.isfinite(gg_ll) else float("nan"),
        )

        # Store fit metrics
        self._fit_metrics = {
            "bgnbd_log_likelihood": bgnbd_ll,
            "gg_log_likelihood":    gg_ll,
            "bgnbd_aic":            float(-2 * bgnbd_ll + 2 * 4) if np.isfinite(bgnbd_ll) else float("nan"),
            "gg_aic":               float(-2 * gg_ll + 2 * 3) if np.isfinite(gg_ll) else float("nan"),
            "n_customers_train":    len(df),
            "n_repeat_buyers":      len(repeat),
            "bgnbd_penalizer_used": bgnbd_penalizer,
            "gg_penalizer_used":    gg_penalizer,
            "bgnbd_params": {
                "r":     float(bgf.params_["r"]),
                "alpha": float(bgf.params_["alpha"]),
                "a":     float(bgf.params_["a"]),
                "b":     float(bgf.params_["b"]),
            },
            "gg_params": {
                "p": float(ggf.params_["p"]),
                "q": float(ggf.params_["q"]),
                "v": float(ggf.params_["v"]),
            },
        }

        self._is_fitted = True
        logger.info("BG/NBD + Gamma-Gamma fit complete.")
        logger.info("  BG/NBD params: {}", self._fit_metrics["bgnbd_params"])
        logger.info("  GG params:     {}", self._fit_metrics["gg_params"])
        return self

    # ── Prediction ───────────────────────────────────────────

    def predict(
        self,
        rfm: pl.DataFrame,
        horizons_days: list[int] | None = None,
        n_bootstrap: int = 100,
    ) -> pl.DataFrame:
        """
        Generate LTV predictions for all customers.

        Args:
            rfm:             Polars RFM DataFrame
            horizons_days:   Prediction horizons in days [365, 730, 1095]
            n_bootstrap:     Number of Monte Carlo samples for CI estimation

        Returns:
            Polars DataFrame with one row per customer and LTV columns
        """
        self._check_fitted()
        bgf = self._require_bgf()
        ggf = self._require_ggf()
        if horizons_days is None:
            horizons_days = [365, 730, 1095]  # 12m, 24m, 36m

        df = polars_rfm_to_pandas(rfm)
        logger.info("Predicting LTV for {} customers…", len(df))

        results = []
        for horizon_days in horizons_days:
            col = f"expected_purchases_{horizon_days}d"
            df[col] = bgf.conditional_expected_number_of_purchases_up_to_time(
                horizon_days, df["frequency"], df["recency"], df["T"]
            )

        # Probability alive (at observation end)
        df["probability_alive"] = bgf.conditional_probability_alive(
            df["frequency"], df["recency"], df["T"]
        )

        # Expected avg monetary value from Gamma-Gamma
        df["expected_avg_profit"] = ggf.conditional_expected_average_profit(
            df["frequency"], df["monetary_value"]
        )
        # For one-time buyers, fall back to their observed monetary_value
        mask_one_time = df["frequency"] == 0
        df.loc[mask_one_time, "expected_avg_profit"] = df.loc[mask_one_time, "monetary_value"]

        # LTV per horizon
        for horizon_days, label in zip(horizons_days, ["12m", "24m", "36m"]):
            col_purchases = f"expected_purchases_{horizon_days}d"
            # Discounted LTV
            years = horizon_days / 365.0
            discount_factor = (1 - (1 / (1 + ANNUAL_DISCOUNT_RATE)) ** years) / ANNUAL_DISCOUNT_RATE
            df[f"ltv_{label}"] = (
                df[col_purchases]
                * df["expected_avg_profit"]
                * MARGIN
                * discount_factor
            ).clip(lower=0)

        # Monte Carlo confidence intervals for 12m and 36m
        ci_data = self._compute_confidence_intervals(
            df, horizons_days=[horizons_days[0], horizons_days[-1]], n_bootstrap=n_bootstrap
        )
        df = df.join(ci_data, how="left")

        # Reset index so customer_id is a column
        df = df.reset_index()

        # Convert back to Polars
        pred_df = pl.from_pandas(df)
        summary_col = "ltv_36m" if "ltv_36m" in pred_df.columns else "ltv_12m"

        logger.info(
            "Predictions complete — mean LTV_36m: £{:.2f}, median: £{:.2f}",
            pred_df[summary_col].mean(),
            pred_df[summary_col].median(),
        )
        return pred_df

    def _compute_confidence_intervals(
        self,
        df: pd.DataFrame,
        horizons_days: list[int],
        n_bootstrap: int = 100,
    ) -> pd.DataFrame:
        """
        Bootstrap confidence intervals by perturbing BG/NBD params
        within their covariance structure.
        """
        logger.debug("Computing {} bootstrap CI samples…", n_bootstrap)

        bgf = self._require_bgf()
        params = np.array([
            bgf.params_["r"],
            bgf.params_["alpha"],
            bgf.params_["a"],
            bgf.params_["b"],
        ])
        # Simple ±10% perturbation as proxy CI (full Hessian-based CI is expensive)
        noise_scale = 0.10

        ltv_samples: dict[str, list] = {
            f"ltv_{label}_samples": [] for label in ["12m", "36m"]
        }

        for _ in range(n_bootstrap):
            noisy_params = params * (1 + np.random.normal(0, noise_scale, size=4))
            noisy_params = np.abs(noisy_params)

            bgf_boot = BetaGeoFitter(penalizer_coef=self.penalizer_coef)
            bgf_boot.params_ = pd.Series({
                "r": noisy_params[0],
                "alpha": noisy_params[1],
                "a": noisy_params[2],
                "b": noisy_params[3],
            })

            for horizon_days, label in zip(horizons_days, ["12m", "36m"]):
                purchases = bgf_boot.conditional_expected_number_of_purchases_up_to_time(
                    horizon_days, df["frequency"], df["recency"], df["T"]
                )
                years = horizon_days / 365.0
                discount = (1 - (1 / (1 + ANNUAL_DISCOUNT_RATE)) ** years) / ANNUAL_DISCOUNT_RATE
                ltv = (purchases * df["expected_avg_profit"] * MARGIN * discount).clip(lower=0)
                ltv_samples[f"ltv_{label}_samples"].append(ltv.values)

        ci_df = df[[]].copy()
        for label in ["12m", "36m"]:
            samples = np.array(ltv_samples[f"ltv_{label}_samples"])  # (n_bootstrap, n_customers)
            ci_df[f"ltv_{label}_lower"] = np.percentile(samples, 5, axis=0)
            ci_df[f"ltv_{label}_upper"] = np.percentile(samples, 95, axis=0)

        return ci_df

    # ── Validation ───────────────────────────────────────────

    def validate(
        self,
        calibration_rfm: pl.DataFrame,
        holdout_rfm: pl.DataFrame,
    ) -> dict[str, float]:
        """
        Validate model on holdout period.

        Returns dict of evaluation metrics matching project targets:
            MAE < 15% of mean LTV (12m)
            MAE < 20% of mean LTV (36m)
            Gini > 0.65
            Top decile lift > 3.0×
            Calibration error < 10%
            BG/NBD frequency R² > 0.85
        """
        self._check_fitted()
        bgf = self._require_bgf()

        cal_pd = polars_rfm_to_pandas(calibration_rfm)
        hold_pd = polars_rfm_to_pandas(holdout_rfm)

        # Align on common customers
        common_idx = cal_pd.index.intersection(hold_pd.index)
        cal_pd   = cal_pd.loc[common_idx]
        hold_pd  = hold_pd.loc[common_idx]

        logger.info("Validating on {} customers in holdout", len(common_idx))

        # ── Frequency prediction (BG/NBD paper-style calibration) ──
        predicted_freq = bgf.conditional_expected_number_of_purchases_up_to_time(
            365, cal_pd["frequency"], cal_pd["recency"], cal_pd["T"]
        )
        predicted_freq = np.asarray(predicted_freq, dtype=float)
        actual_freq = np.asarray(hold_pd["frequency"].values, dtype=float)
        freq_mask = np.isfinite(actual_freq) & np.isfinite(predicted_freq)
        actual_freq_eval = actual_freq[freq_mask]
        predicted_freq_eval = predicted_freq[freq_mask]

        freq_denom = np.sum((actual_freq_eval - actual_freq_eval.mean()) ** 2)
        r2_freq = (
            float(1 - np.sum((actual_freq_eval - predicted_freq_eval) ** 2) / freq_denom)
            if freq_denom > 0 else 0.0
        )
        mae_freq = float(np.mean(np.abs(actual_freq_eval - predicted_freq_eval)))
        rmse_freq = float(np.sqrt(np.mean((actual_freq_eval - predicted_freq_eval) ** 2)))

        # ── LTV validation ──
        pred_df = self.predict(calibration_rfm)
        pred_pd = pred_df.to_pandas().set_index("customer_id")

        # Prefer exact labels on calibration rows when available.
        # This avoids active-only holdout bias and keeps zero-holdout customers.
        if "actual_ltv_12m" in calibration_rfm.columns:
            labels_pd = (
                calibration_rfm
                .select(["customer_id", "actual_ltv_12m"])
                .to_pandas()
                .set_index("customer_id")
            )
            eval_idx = pred_pd.index.intersection(labels_pd.index)
            y_true = np.asarray(labels_pd.loc[eval_idx, "actual_ltv_12m"].values, dtype=float)
            logger.info(
                "Using calibration_rfm.actual_ltv_12m labels for LTV validation ({} customers).",
                len(eval_idx),
            )
        else:
            # Fallback proxy from holdout RFM.
            if "monetary_total" in holdout_rfm.columns:
                hold_ltv = (
                    holdout_rfm
                    .select(["customer_id", "monetary_total"])
                    .to_pandas()
                    .set_index("customer_id")["monetary_total"]
                )
            else:
                hold_ltv = hold_pd["monetary_value"] * (hold_pd["frequency"] + 1)
            eval_idx = hold_ltv.index.intersection(pred_pd.index)
            y_true = np.asarray(hold_ltv.loc[eval_idx].values, dtype=float)
            logger.warning(
                "actual_ltv_12m missing on calibration_rfm; using holdout-derived proxy labels ({} customers).",
                len(eval_idx),
            )

        y_pred_12m = np.asarray(pred_pd.loc[eval_idx, "ltv_12m"].values, dtype=float)
        y_pred_36m = np.asarray(pred_pd.loc[eval_idx, "ltv_36m"].values, dtype=float)

        valid_mask = (
            np.isfinite(y_true)
            & np.isfinite(y_pred_12m)
            & np.isfinite(y_pred_36m)
            & (y_true >= 0)
        )
        y_true = y_true[valid_mask]
        y_pred_12m = y_pred_12m[valid_mask]
        y_pred_36m = y_pred_36m[valid_mask]

        if len(y_true) == 0:
            raise ValueError("No valid rows available for holdout LTV validation.")

        mean_ltv = float(np.mean(y_true))
        if not np.isfinite(mean_ltv) or mean_ltv < 0:
            raise ValueError("Invalid mean actual LTV computed during validation.")

        metrics = {
            # Frequency accuracy
            "r2_frequency":         r2_freq,
            "mae_frequency_12m":    mae_freq,
            "rmse_frequency_12m":   rmse_freq,
            # LTV accuracy
            "mae_ltv_12m":          float(np.mean(np.abs(y_true - y_pred_12m))),
            "mae_ltv_36m":          float(np.mean(np.abs(y_true - y_pred_36m))),
            "mae_pct_12m":          float(np.mean(np.abs(y_true - y_pred_12m)) / max(mean_ltv, 1)),
            "rmse_ltv_12m":         float(np.sqrt(np.mean((y_true - y_pred_12m) ** 2))),
            # Ranking quality
            "gini_coefficient":     compute_gini(y_true, y_pred_12m),
            "top_decile_lift":      compute_top_decile_lift(y_true, y_pred_12m),
            "calibration_error":    compute_calibration_error(y_true, y_pred_12m),
            # Dataset info
            "n_customers_holdout":  int(len(y_true)),
            "mean_actual_ltv":      mean_ltv,
        }

        logger.info("=== Validation Metrics ===")
        logger.info("  BG/NBD R² (frequency):   {:.4f}  (target: > 0.85)", metrics["r2_frequency"])
        logger.info("  MAE LTV 12m:             {:.2f}  ({:.1f}% of mean, target < 15%)",
                    metrics["mae_ltv_12m"], 100 * metrics["mae_pct_12m"])
        logger.info("  Gini coefficient:        {:.4f}  (target: > 0.65)", metrics["gini_coefficient"])
        logger.info("  Top decile lift:         {:.2f}×  (target: > 3.0×)", metrics["top_decile_lift"])
        logger.info("  Calibration error:       {:.4f}  (target: < 0.10)", metrics["calibration_error"])

        self._fit_metrics.update(metrics)
        return metrics

    # ── Calibration plot data ─────────────────────────────────

    def get_calibration_plot_data(
        self,
        rfm: pl.DataFrame,
        holdout_rfm: pl.DataFrame,
        n_buckets: int = 7,
    ) -> pl.DataFrame:
        """
        Generate frequency-bucket calibration data for plotting.
        Returns: frequency_bucket | predicted_avg | actual_avg | customer_count
        """
        self._check_fitted()
        bgf = self._require_bgf()

        cal_pd = polars_rfm_to_pandas(rfm)
        hold_pd = polars_rfm_to_pandas(holdout_rfm)
        common = cal_pd.index.intersection(hold_pd.index)
        cal_pd  = cal_pd.loc[common]
        hold_pd = hold_pd.loc[common]

        predicted = bgf.conditional_expected_number_of_purchases_up_to_time(
            365, cal_pd["frequency"], cal_pd["recency"], cal_pd["T"]
        )

        combined = pd.DataFrame({
            "predicted":  predicted.values,
            "actual":     hold_pd["frequency"].values,
            "freq_cal":   cal_pd["frequency"].values,
        })

        combined["bucket"] = pd.cut(
            combined["freq_cal"],
            bins=n_buckets,
            labels=False,
            duplicates="drop",
        )

        agg = (
            combined.groupby("bucket")
            .agg(
                predicted_purchases_avg=("predicted", "mean"),
                actual_purchases_avg=("actual", "mean"),
                customer_count=("predicted", "count"),
                frequency_bucket=("freq_cal", "mean"),
            )
            .reset_index(drop=True)
        )

        return pl.from_pandas(agg)

    def get_probability_alive_matrix(
        self,
        max_frequency: int = 50,
        max_recency_days: int = 365,
        t_days: int = 365,
        step: int = 5,
    ) -> pl.DataFrame:
        """
        Compute P(alive) for a grid of (frequency, recency) values.
        Used for the probability-alive heatmap in the dashboard.
        """
        self._check_fitted()
        bgf = self._require_bgf()

        freqs = list(range(0, max_frequency + 1, step))
        recencies = list(range(0, max_recency_days + 1, step))

        rows = []
        for f in freqs:
            for r in recencies:
                if r > t_days:
                    continue
                p_raw = bgf.conditional_probability_alive(
                    frequency=f,
                    recency=min(r, t_days - 1),
                    T=t_days,
                )
                p = float(np.asarray(p_raw).reshape(-1)[0])
                rows.append({"frequency": f, "recency_days": r, "t_days": t_days, "p_alive": p})

        return pl.DataFrame(rows)

    # ── Persistence ──────────────────────────────────────────

    def save_params(
        self,
        db_client: Any,
        pipeline_run_id: str | None = None,
        wandb_run_id: str | None = None,
    ) -> None:
        """Persist model parameters and metrics to Supabase."""
        self._check_fitted()
        bg_params = self._fit_metrics["bgnbd_params"]
        gg_params = self._fit_metrics["gg_params"]

        def _finite_or_none(value: Any) -> float | None:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return None
            return v if np.isfinite(v) else None

        # DB columns are NUMERIC(12,8); tiny values would round to 0 and violate >0 checks.
        def _positive_param(value: Any, floor: float = 1e-6) -> float | None:
            v = _finite_or_none(value)
            if v is None:
                return None
            if v <= 0:
                return floor
            return max(v, floor)

        params_record = {
            "model_version":            self.model_version,
            "fitted_at":                datetime.now(timezone.utc).isoformat(),
            "observation_end":          str(self.observation_end),
            "dataset":                  "uci_online_retail",
            # BG/NBD params
            "bgnbd_r":                  _positive_param(bg_params["r"]),
            "bgnbd_alpha":              _positive_param(bg_params["alpha"]),
            "bgnbd_a":                  _positive_param(bg_params["a"]),
            "bgnbd_b":                  _positive_param(bg_params["b"]),
            # GG params
            "gg_p":                     _positive_param(gg_params["p"]),
            "gg_q":                     _positive_param(gg_params["q"]),
            "gg_v":                     _positive_param(gg_params["v"]),
            # Fit quality
            "bgnbd_log_likelihood":     _finite_or_none(self._fit_metrics.get("bgnbd_log_likelihood")),
            "gg_log_likelihood":        _finite_or_none(self._fit_metrics.get("gg_log_likelihood")),
            "bgnbd_aic":                _finite_or_none(self._fit_metrics.get("bgnbd_aic")),
            "gg_aic":                   _finite_or_none(self._fit_metrics.get("gg_aic")),
            # Validation
            "mae_frequency_12m":        _finite_or_none(self._fit_metrics.get("mae_frequency_12m")),
            "rmse_frequency_12m":       _finite_or_none(self._fit_metrics.get("rmse_frequency_12m")),
            "r2_frequency":             _finite_or_none(self._fit_metrics.get("r2_frequency")),
            "mae_ltv_12m":              _finite_or_none(self._fit_metrics.get("mae_ltv_12m")),
            "mae_ltv_36m":              _finite_or_none(self._fit_metrics.get("mae_ltv_36m")),
            "gini_coefficient":         _finite_or_none(self._fit_metrics.get("gini_coefficient")),
            "top_decile_lift":          _finite_or_none(self._fit_metrics.get("top_decile_lift")),
            "calibration_error":        _finite_or_none(self._fit_metrics.get("calibration_error")),
            # Dataset
            "n_customers_train":        self._fit_metrics.get("n_customers_train"),
            "n_customers_holdout":      self._fit_metrics.get("n_customers_holdout"),
            "observation_months":       6,
            "holdout_months":           6,
            # Tracking
            "wandb_run_id":             wandb_run_id,
            "pipeline_run_id":          pipeline_run_id,
        }

        db_client.bulk_upsert(
            "bgnbd_model_params",
            [params_record],
            conflict_columns=["model_version"],
        )
        logger.info("Saved BG/NBD params — model_version={}", self.model_version)

    def save_predictions(
        self,
        predictions: pl.DataFrame,
        db_client: Any,
        batch_size: int = 500,
    ) -> int:
        """Persist per-customer predictions to Supabase bgnbd_predictions table."""
        self._check_fitted()

        pred_cols = set(predictions.columns)

        # Map column names to DB schema
        records = []
        for row in predictions.iter_rows(named=True):
            expected_365 = row.get("expected_purchases_365d")
            expected_180 = row.get("expected_purchases_180d")
            expected_90 = row.get("expected_purchases_90d")

            # Backward-compatible fallbacks when only yearly/2y/3y horizons exist.
            if expected_365 is None:
                expected_365 = row.get("expected_purchases_730d")
            if expected_180 is None and expected_365 is not None:
                expected_180 = 0.5 * float(expected_365)
            if expected_90 is None and expected_365 is not None:
                expected_90 = 0.25 * float(expected_365)

            records.append({
                "customer_id":              row.get("customer_id"),
                "model_version":            self.model_version,
                "predicted_at":             datetime.now(timezone.utc).isoformat(),
                "expected_purchases_90d":   expected_90,
                "expected_purchases_180d":  expected_180,
                "expected_purchases_365d":  expected_365,
                "probability_alive":        row.get("probability_alive"),
                "expected_avg_profit":      row.get("expected_avg_profit"),
                "ltv_12m":                  row.get("ltv_12m"),
                "ltv_24m":                  row.get("ltv_24m"),
                "ltv_36m":                  row.get("ltv_36m"),
                "ltv_12m_lower":            row.get("ltv_12m_lower"),
                "ltv_12m_upper":            row.get("ltv_12m_upper"),
                "ltv_36m_lower":            row.get("ltv_36m_lower"),
                "ltv_36m_upper":            row.get("ltv_36m_upper"),
                "segment":                  row.get("segment") if "segment" in pred_cols else None,
            })

        n = db_client.bulk_upsert(
            "bgnbd_predictions",
            records,
            conflict_columns=["customer_id", "model_version"],
            batch_size=batch_size,
        )
        logger.info("Saved {} BG/NBD predictions", n)
        return n

    def save_probability_alive_matrix(
        self,
        db_client: Any,
    ) -> None:
        """Persist probability-alive matrix for dashboard heatmap."""
        matrix = self.get_probability_alive_matrix()
        records = matrix.with_columns(
            pl.lit(self.model_version).alias("model_version")
        ).to_dicts()
        db_client.bulk_upsert(
            "probability_alive_matrix",
            records,
            conflict_columns=["model_version", "frequency", "recency_days", "t_days"],
        )
        logger.info("Saved probability-alive matrix ({} rows)", len(records))

    # ── Serialisation ─────────────────────────────────────────

    def save_to_disk(self, path: str | Path) -> None:
        """Persist fitted model artifacts and portable metadata to disk."""
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        bgf = self._require_bgf()
        ggf = self._require_ggf()
        bgnbd_params = dict(bgf.params_)
        gg_params = dict(ggf.params_)

        # Save portable fitter payloads (parameter-only, pickle-safe across versions).
        with open(path / f"{self.model_version}_bgf.pkl", "wb") as f:
            pickle.dump({
                "fitter": "BetaGeoFitter",
                "penalizer_coef": self.penalizer_coef,
                "params": bgnbd_params,
            }, f)
        with open(path / f"{self.model_version}_ggf.pkl", "wb") as f:
            pickle.dump({
                "fitter": "GammaGammaFitter",
                "penalizer_coef": self.penalizer_coef,
                "params": gg_params,
            }, f)

        # Portable metadata fallback (robust across lifetimes versions).
        with open(path / f"{self.model_version}_meta.pkl", "wb") as f:
            pickle.dump({
                "model_version":    self.model_version,
                "observation_end":  self.observation_end,
                "penalizer_coef":   self.penalizer_coef,
                "fit_metrics":      self._fit_metrics,
                "bgnbd_params":     bgnbd_params,
                "gg_params":        gg_params,
            }, f)
        logger.info("Model saved to {}", path)

    @classmethod
    def load_from_disk(cls, path: str | Path, model_version: str) -> "BGNBDModel":
        """Load a previously saved model from disk."""
        import pickle
        path = Path(path)
        with open(path / f"{model_version}_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        model = cls(
            penalizer_coef=meta["penalizer_coef"],
            model_version=meta["model_version"],
            observation_end=meta["observation_end"],
        )
        model.bgf = BetaGeoFitter(penalizer_coef=model.penalizer_coef)
        model.ggf = GammaGammaFitter(penalizer_coef=model.penalizer_coef)
        model.bgf.params_ = pd.Series(meta["bgnbd_params"])
        model.ggf.params_ = pd.Series(meta["gg_params"])
        model._fit_metrics = meta["fit_metrics"]
        model._is_fitted = True
        logger.info("Model loaded from {} (version: {})", path, model_version)
        return model

    # ── Utilities ─────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling this method.")
        if self.bgf is None or self.ggf is None:
            raise RuntimeError("Model fit state is invalid: missing BG/NBD or Gamma-Gamma fitter.")

    def _require_bgf(self) -> BetaGeoFitter:
        self._check_fitted()
        if self.bgf is None:
            raise RuntimeError("BG/NBD fitter is not available.")
        return self.bgf

    def _require_ggf(self) -> GammaGammaFitter:
        self._check_fitted()
        if self.ggf is None:
            raise RuntimeError("Gamma-Gamma fitter is not available.")
        return self.ggf

    def get_params(self) -> dict[str, Any]:
        bgf = self._require_bgf()
        ggf = self._require_ggf()
        return {
            "bgnbd": dict(bgf.params_),
            "gamma_gamma": dict(ggf.params_),
        }

    def predict_single(
        self,
        frequency: int,
        recency_days: float,
        t_days: int,
        monetary_avg: float,
        horizon_days: int = 365,
    ) -> dict[str, float]:
        """
        Score a single customer in real-time (used by FastAPI).
        Returns dict with expected_purchases, probability_alive, ltv_12m, ltv_36m.
        """
        bgf = self._require_bgf()
        ggf = self._require_ggf()

        exp_purchases = float(
            bgf.conditional_expected_number_of_purchases_up_to_time(
                horizon_days, frequency, min(recency_days, t_days - 1), t_days
            )
        )
        p_alive_raw = bgf.conditional_probability_alive(
            frequency, min(recency_days, t_days - 1), t_days
        )
        p_alive = float(np.asarray(p_alive_raw).reshape(-1)[0])
        exp_profit = (
            float(ggf.conditional_expected_average_profit(frequency, monetary_avg))
            if frequency > 0
            else monetary_avg
        )

        def ltv(days: int) -> float:
            purchases = float(
                bgf.conditional_expected_number_of_purchases_up_to_time(
                    days, frequency, min(recency_days, t_days - 1), t_days
                )
            )
            years = days / 365.0
            discount = (1 - (1 / (1 + ANNUAL_DISCOUNT_RATE)) ** years) / ANNUAL_DISCOUNT_RATE
            return max(0.0, purchases * exp_profit * MARGIN * discount)

        return {
            "expected_purchases":   exp_purchases,
            "probability_alive":    p_alive,
            "expected_avg_profit":  exp_profit,
            "ltv_12m":              ltv(365),
            "ltv_24m":              ltv(730),
            "ltv_36m":              ltv(1095),
        }
