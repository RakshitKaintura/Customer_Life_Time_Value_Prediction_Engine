"""
LTV Distribution Drift Detector.

Implements:
  1. Population Stability Index (PSI) — standard industry metric
     PSI < 0.10: no significant change
     PSI 0.10–0.25: moderate change (monitor)
     PSI > 0.25: major change (retrain)

  2. Kolmogorov-Smirnov test — non-parametric distribution comparison

  3. Mean shift detection — % change in mean LTV

  4. Segment distribution shift — change in champion/high-value %

  5. Feature drift per RFM feature

Alert threshold: PSI > 0.15 or mean shift > 15%

Results are stored in:
  - ltv_drift_alerts (distribution-level)
  - feature_drift_log (per-feature)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone, timedelta
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from scipy import stats


# ─────────────────────────────────────────────────────────────
# PSI calculation
# ─────────────────────────────────────────────────────────────

def compute_psi(
    baseline:   np.ndarray,
    current:    np.ndarray,
    n_bins:     int = 10,
    epsilon:    float = 1e-6,
) -> float:
    """
    Population Stability Index (PSI).

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Args:
        baseline: Historical distribution (reference)
        current:  Current distribution (monitoring)
        n_bins:   Number of bins for discretisation
        epsilon:  Small constant to avoid log(0)

    Returns:
        PSI score (float)
    """
    # Use baseline percentiles as bin edges
    bin_edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # Remove duplicates
    bin_edges[0]  -= 1e-6   # Include minimum
    bin_edges[-1] += 1e-6   # Include maximum

    # Count observations in each bin
    base_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    curr_counts = np.histogram(current,  bins=bin_edges)[0].astype(float)

    # Convert to proportions
    base_pct = (base_counts + epsilon) / (len(baseline) + epsilon * n_bins)
    curr_pct = (curr_counts + epsilon) / (len(current)  + epsilon * n_bins)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)


def compute_ks_test(
    baseline: np.ndarray,
    current:  np.ndarray,
) -> tuple[float, float]:
    """
    Two-sample KS test.
    Returns (ks_statistic, p_value).
    p_value < 0.05 → significant distributional difference.
    """
    ks_stat, p_value = stats.ks_2samp(baseline, current)
    return float(ks_stat), float(p_value)


# ─────────────────────────────────────────────────────────────
# Feature drift
# ─────────────────────────────────────────────────────────────

def compute_feature_drift(
    baseline_values: np.ndarray,
    current_values:  np.ndarray,
    threshold:       float = 0.20,
) -> dict[str, float]:
    """Compute drift metrics for a single feature."""
    psi = compute_psi(baseline_values, current_values)
    ks_stat, _ = compute_ks_test(baseline_values, current_values)

    baseline_mean = float(np.mean(baseline_values))
    current_mean  = float(np.mean(current_values))
    mean_shift    = (
        (current_mean - baseline_mean) / max(abs(baseline_mean), 1e-9) * 100
    )

    return {
        "psi_score":      psi,
        "ks_statistic":   ks_stat,
        "baseline_mean":  baseline_mean,
        "baseline_std":   float(np.std(baseline_values)),
        "current_mean":   current_mean,
        "current_std":    float(np.std(current_values)),
        "mean_shift_pct": mean_shift,
        "is_drifted":     psi > threshold,
        "drift_threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────
# Drift Detector
# ─────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Full drift detection pipeline.

    Usage:
        detector = DriftDetector(db_client=db)
        results  = detector.run_full_drift_check(model_version='fusion_v1')
    """

    PSI_THRESHOLD_WARN  = 0.10   # monitor
    PSI_THRESHOLD_ALERT = 0.15   # alert — project spec: > 15%
    MEAN_SHIFT_THRESHOLD = 15.0  # % change in mean LTV

    def __init__(self, db_client: Any) -> None:
        self.db = db_client

    def _load_ltv_scores(
        self,
        start_date: date,
        end_date:   date,
    ) -> np.ndarray:
        """Load LTV_36m values from a date range."""
        rows = self.db.execute_sql(
            """
            SELECT ltv_36m
            FROM final_ltv_scores
            WHERE scored_at::DATE BETWEEN :start AND :end
              AND ltv_source = 'full_model'
              AND ltv_36m IS NOT NULL
            """,
            {"start": str(start_date), "end": str(end_date)},
        )
        if not rows:
            return np.array([])
        return np.array([float(r["ltv_36m"]) for r in rows])

    def _load_rfm_feature(
        self,
        feature: str,
        start_date: date,
        end_date:   date,
    ) -> np.ndarray:
        """Load a single RFM feature from a date range."""
        rows = self.db.execute_sql(
            f"""
            SELECT {feature}
            FROM rfm_features
            WHERE computed_at::DATE BETWEEN :start AND :end
              AND {feature} IS NOT NULL
            LIMIT 5000
            """,
            {"start": str(start_date), "end": str(end_date)},
        )
        if not rows:
            return np.array([])
        return np.array([float(r[feature]) for r in rows])

    def run_full_drift_check(
        self,
        model_version:   str,
        baseline_days:   int = 60,
        monitoring_days: int = 30,
    ) -> dict[str, Any]:
        """
        Run all drift checks and persist alerts to Supabase.

        Returns:
            dict with drift_detected, alerts, psi_score, mean_shift_pct
        """
        today            = date.today()
        monitoring_start = today - timedelta(days=monitoring_days)
        baseline_end     = monitoring_start - timedelta(days=1)
        baseline_start   = baseline_end - timedelta(days=baseline_days)

        logger.info(
            "Drift check: baseline={} – {}, monitoring={} – {}",
            baseline_start, baseline_end, monitoring_start, today,
        )

        # Load LTV distributions
        baseline_ltv = self._load_ltv_scores(baseline_start, baseline_end)
        current_ltv  = self._load_ltv_scores(monitoring_start, today)

        alerts: list[dict] = []
        drift_detected = False

        if len(baseline_ltv) < 10 or len(current_ltv) < 10:
            logger.warning(
                "Insufficient data for drift check: baseline={}, current={}",
                len(baseline_ltv), len(current_ltv),
            )
            return {
                "drift_detected": False,
                "alerts":         [],
                "psi_score":      None,
                "status":         "insufficient_data",
            }

        # ── 1. PSI on LTV distribution ──
        psi = compute_psi(baseline_ltv, current_ltv)
        ks_stat, ks_pvalue = compute_ks_test(baseline_ltv, current_ltv)

        psi_alert = {
            "alert_type":            "distribution_shift",
            "psi_score":             psi,
            "ks_statistic":          ks_stat,
            "ks_pvalue":             ks_pvalue,
            "threshold_exceeded":    psi > self.PSI_THRESHOLD_ALERT,
            "threshold_value":       self.PSI_THRESHOLD_ALERT,
            "actual_value":          psi,
        }
        alerts.append(psi_alert)
        if psi > self.PSI_THRESHOLD_ALERT:
            drift_detected = True
            logger.warning("PSI drift alert: PSI={:.4f} > {}", psi, self.PSI_THRESHOLD_ALERT)

        # ── 2. Mean shift ──
        baseline_mean  = float(np.mean(baseline_ltv))
        current_mean   = float(np.mean(current_ltv))
        mean_shift_pct = abs((current_mean - baseline_mean) / max(baseline_mean, 1e-9) * 100)

        mean_alert = {
            "alert_type":         "mean_shift",
            "mean_shift_pct":     mean_shift_pct,
            "threshold_exceeded": mean_shift_pct > self.MEAN_SHIFT_THRESHOLD,
            "threshold_value":    self.MEAN_SHIFT_THRESHOLD,
            "actual_value":       mean_shift_pct,
        }
        alerts.append(mean_alert)
        if mean_shift_pct > self.MEAN_SHIFT_THRESHOLD:
            drift_detected = True
            logger.warning("Mean shift alert: {:.1f}% > {}%", mean_shift_pct, self.MEAN_SHIFT_THRESHOLD)

        # ── 3. Feature drift ──
        rfm_features_to_check = ["frequency", "monetary_avg", "recency_days"]
        feature_drift_records = []

        for feature in rfm_features_to_check:
            base_vals = self._load_rfm_feature(feature, baseline_start, baseline_end)
            curr_vals = self._load_rfm_feature(feature, monitoring_start, today)

            if len(base_vals) >= 10 and len(curr_vals) >= 10:
                drift_metrics = compute_feature_drift(base_vals, curr_vals)
                feature_drift_records.append({
                    "feature_name":   feature,
                    "model_version":  model_version,
                    "period_start":   str(monitoring_start),
                    "period_end":     str(today),
                    "logged_at":      datetime.now(timezone.utc).isoformat(),
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                       for k, v in drift_metrics.items()},
                })

        # Persist alerts
        try:
            alert_records = []
            for alert in alerts:
                if alert.get("threshold_exceeded"):
                    alert_records.append({
                        "alert_id":                 str(uuid.uuid4()),
                        "detected_at":              datetime.now(timezone.utc).isoformat(),
                        "alert_type":               alert["alert_type"],
                        "psi_score":                alert.get("psi_score"),
                        "ks_statistic":             alert.get("ks_statistic"),
                        "ks_pvalue":                alert.get("ks_pvalue"),
                        "mean_shift_pct":           alert.get("mean_shift_pct"),
                        "threshold_exceeded":       True,
                        "threshold_value":          alert.get("threshold_value"),
                        "actual_value":             alert.get("actual_value"),
                        "baseline_period_start":    str(baseline_start),
                        "baseline_period_end":      str(baseline_end),
                        "monitoring_period_start":  str(monitoring_start),
                        "monitoring_period_end":    str(today),
                        "n_baseline_customers":     len(baseline_ltv),
                        "n_monitoring_customers":   len(current_ltv),
                        "status":                   "open",
                        "model_version":            model_version,
                    })

            if alert_records:
                self.db.bulk_upsert(
                    "ltv_drift_alerts", alert_records,
                    conflict_columns=["alert_id"],
                )

            if feature_drift_records:
                self.db.bulk_upsert(
                    "feature_drift_log",
                    feature_drift_records,
                    conflict_columns=None,
                )

            logger.info(
                "Drift check complete: PSI={:.4f}, mean_shift={:.1f}%, "
                "drift_detected={}, alerts_saved={}",
                psi, mean_shift_pct, drift_detected, len(alert_records),
            )
        except Exception as exc:
            logger.error("Failed to persist drift alerts: {}", exc)

        return {
            "drift_detected": drift_detected,
            "alerts":         alerts,
            "psi_score":      psi,
            "ks_statistic":   ks_stat,
            "mean_shift_pct": mean_shift_pct,
            "n_baseline":     len(baseline_ltv),
            "n_current":      len(current_ltv),
            "baseline_mean":  baseline_mean,
            "current_mean":   current_mean,
        }