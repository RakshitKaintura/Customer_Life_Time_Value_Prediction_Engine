"""
XGBoost Stacking Meta-Learner — Fusion Layer.

Combines BG/NBD + Gamma-Gamma and Transformer predictions
into a single calibrated LTV score.

Design:
  Level-0 predictions (features for meta-learner):
    - bgnbd_ltv_12m, bgnbd_ltv_36m
    - transformer_ltv_12m, transformer_ltv_36m
    - probability_alive
    - transaction_count      → meta-learner trusts BG/NBD less when low
    - purchase_variance      → meta-learner trusts Transformer more when high
    - customer_age_days
    - acquisition_channel    (encoded)
    - recency_days
    - monetary_avg

  Level-1 XGBoost regressor trained on validation set predictions
  Output: final LTV_12m and LTV_36m

  Optuna tunes: n_estimators, max_depth, lr, subsample, colsample_bytree

Design decision:
  We train on the VALIDATION set of the original calibration split.
  This ensures the meta-learner never sees training data from level-0 models,
  preventing data leakage.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed. Run: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed. Run: pip install shap")


# ─────────────────────────────────────────────────────────────
# Feature engineering for meta-learner
# ─────────────────────────────────────────────────────────────

META_FEATURES = [
    "bgnbd_ltv_12m",
    "bgnbd_ltv_36m",
    "transformer_ltv_12m",
    "transformer_ltv_36m",
    "probability_alive",
    "frequency",
    "monetary_avg",
    "recency_days",
    "t_days",
    "purchase_variance",
    "orders_count",
    "avg_days_between_orders",
    "unique_categories",
]

CATEGORICAL_FEATURES: list[str] = []   # no categoricals in base set


def build_meta_features(
    bgnbd_preds: pl.DataFrame,
    transformer_preds: pl.DataFrame,
    rfm_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Join BG/NBD predictions, Transformer predictions, and RFM features
    into a single meta-learner feature matrix.

    All three DataFrames must have a 'customer_id' column.

    Returns Polars DataFrame with columns: customer_id + META_FEATURES
    """
    # Rename columns to avoid conflicts
    bgnbd = bgnbd_preds.select([
        "customer_id",
        pl.col("ltv_12m").alias("bgnbd_ltv_12m"),
        pl.col("ltv_36m").alias("bgnbd_ltv_36m"),
        pl.col("probability_alive"),
        pl.col("expected_purchases_365d").alias("expected_purchases_12m")
            if "expected_purchases_365d" in bgnbd_preds.columns
            else pl.lit(None).cast(pl.Float64).alias("expected_purchases_12m"),
    ])

    transformer = transformer_preds.select([
        "customer_id",
        pl.col("ltv_12m").alias("transformer_ltv_12m"),
        pl.col("ltv_36m").alias("transformer_ltv_36m"),
    ])

    rfm = rfm_df.select([
        "customer_id",
        "frequency",
        "monetary_avg",
        "recency_days",
        "t_days",
        pl.col("purchase_variance").fill_null(0.0),
        pl.col("orders_count").fill_null(1),
        pl.col("avg_days_between_orders").fill_null(30.0),
        pl.col("unique_categories").fill_null(1),
    ])

    # Join all
    meta = (
        bgnbd
        .join(transformer, on="customer_id", how="inner")
        .join(rfm,         on="customer_id", how="left")
    )

    # Fill any remaining nulls
    fill_zero_cols = [
        "bgnbd_ltv_12m", "bgnbd_ltv_36m",
        "transformer_ltv_12m", "transformer_ltv_36m",
        "probability_alive", "frequency", "monetary_avg",
        "recency_days", "t_days", "purchase_variance",
        "orders_count", "avg_days_between_orders", "unique_categories",
    ]
    for col in fill_zero_cols:
        if col in meta.columns:
            meta = meta.with_columns(pl.col(col).fill_null(0.0))

    logger.info(
        "Meta-feature matrix: {} customers × {} features",
        len(meta), len(meta.columns) - 1,  # -1 for customer_id
    )
    return meta


# ─────────────────────────────────────────────────────────────
# XGBoost Meta-Learner
# ─────────────────────────────────────────────────────────────

class XGBoostMetaLearner:
    """
    XGBoost stacking regressor for LTV fusion.

    Trained on held-out validation set predictions from
    BG/NBD and Transformer models.

    Predicts:
        - ltv_12m  (primary target for CAC bidding)
        - ltv_36m  (primary target for segmentation)
    """

    def __init__(
        self,
        model_version: str = "fusion_v1",
        xgb_params: dict | None = None,
    ) -> None:
        if not XGB_AVAILABLE:
            raise ImportError("xgboost is required. Run: pip install xgboost")

        self.model_version = model_version
        self.xgb_params = xgb_params or {
            "n_estimators":       200,
            "max_depth":          4,
            "learning_rate":      0.05,
            "subsample":          0.8,
            "colsample_bytree":   0.8,
            "min_child_weight":   5,
            "reg_alpha":          0.1,
            "reg_lambda":         1.0,
            "random_state":       42,
            "n_jobs":             -1,
            "objective":          "reg:squarederror",
        }

        self.model_12m: xgb.XGBRegressor | None = None
        self.model_36m: xgb.XGBRegressor | None = None
        self._feature_names: list[str] = []
        self._is_fitted = False
        self._train_metrics: dict[str, Any] = {}
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._target_transform: Callable[[np.ndarray], np.ndarray] | None = None
        self._inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None
        self._transform_name: str | None = None

    # ── Fitting ──────────────────────────────────────────────

    def fit(
        self,
        meta_features: pl.DataFrame,
        targets: pl.DataFrame,
        eval_set_features: pl.DataFrame | None = None,
        eval_set_targets: pl.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
        eval_sample_weight: np.ndarray | None = None,
        target_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "XGBoostMetaLearner":
        """
        Fit two XGBoost regressors: one for LTV_12m, one for LTV_36m.

        Args:
            meta_features:      Polars DataFrame with META_FEATURES columns
            targets:            Polars DataFrame with customer_id + actual_ltv_12m + actual_ltv_36m
            eval_set_features:  Optional validation set for early stopping
            eval_set_targets:   Optional validation targets
        """
        # Join meta features with targets
        df = meta_features.join(
            targets.select(["customer_id", "actual_ltv_12m",
                            pl.col("actual_ltv_12m").alias("actual_ltv_36m")
                            if "actual_ltv_36m" not in targets.columns
                            else "actual_ltv_36m"]),
            on="customer_id",
            how="inner",
        )

        feature_cols = [c for c in META_FEATURES if c in df.columns]
        self._feature_names = feature_cols

        self._target_transform = target_transform
        self._inverse_transform = inverse_transform
        if target_transform is np.log1p and inverse_transform is np.expm1:
            self._transform_name = "log1p"
        else:
            self._transform_name = None

        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y_12m_raw = df["actual_ltv_12m"].to_numpy().astype(np.float32)
        y_36m_raw = (
            df["actual_ltv_36m"].to_numpy().astype(np.float32)
            if "actual_ltv_36m" in df.columns
            else y_12m_raw * 2.5
        )

        if target_transform is not None:
            y_12m = target_transform(y_12m_raw)
            y_36m = target_transform(y_36m_raw)
        else:
            y_12m = y_12m_raw
            y_36m = y_36m_raw

        logger.info(
            "Training XGBoost meta-learner: {} samples, {} features",
            len(X), len(feature_cols),
        )

        # ── LTV 12m model ──
        self.model_12m = xgb.XGBRegressor(**self.xgb_params)

        if eval_set_features is not None and eval_set_targets is not None:
            eval_df = eval_set_features.join(
                eval_set_targets.select(["customer_id", "actual_ltv_12m"]),
                on="customer_id",
                how="inner",
            )
            X_val = eval_df.select(feature_cols).to_numpy().astype(np.float32)
            y_val_raw = eval_df["actual_ltv_12m"].to_numpy().astype(np.float32)
            y_val = target_transform(y_val_raw) if target_transform is not None else y_val_raw
            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            if eval_sample_weight is not None:
                fit_kwargs["sample_weight_eval_set"] = [eval_sample_weight]
            self.model_12m.fit(X, y_12m, **fit_kwargs)
        else:
            self.model_12m.fit(X, y_12m, sample_weight=sample_weight)

        # ── LTV 36m model ──
        self.model_36m = xgb.XGBRegressor(**self.xgb_params)
        self.model_36m.fit(X, y_36m, sample_weight=sample_weight)

        # Training metrics
        pred_12m_train = self.model_12m.predict(X)
        pred_36m_train = self.model_36m.predict(X)
        if inverse_transform is not None:
            pred_12m_train = inverse_transform(np.clip(pred_12m_train, 0, None))
            pred_36m_train = inverse_transform(np.clip(pred_36m_train, 0, None))
        else:
            pred_12m_train = np.clip(pred_12m_train, 0, None)
            pred_36m_train = np.clip(pred_36m_train, 0, None)

        self._train_metrics = {
            "n_train":        len(X),
            "n_features":     len(feature_cols),
            "mae_12m_train":  float(np.mean(np.abs(y_12m_raw - pred_12m_train))),
            "mae_36m_train":  float(np.mean(np.abs(y_36m_raw - pred_36m_train))),
            "mean_ltv_12m":   float(y_12m_raw.mean()),
        }

        self._is_fitted = True
        logger.info(
            "Meta-learner trained — MAE_12m={:.2f}  MAE_36m={:.2f}",
            self._train_metrics["mae_12m_train"],
            self._train_metrics["mae_36m_train"],
        )
        return self

    # ── Prediction ───────────────────────────────────────────

    def predict(
        self,
        meta_features: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Generate final ensemble LTV predictions.

        Returns Polars DataFrame with:
            customer_id | ltv_12m | ltv_24m | ltv_36m |
            meta_weight_bgnbd | meta_weight_transformer
        """
        model_12m, model_36m = self._require_models()

        feature_cols = [c for c in self._feature_names if c in meta_features.columns]
        X = meta_features.select(feature_cols).to_numpy().astype(np.float32)

        pred_12m = model_12m.predict(X)
        pred_36m = model_36m.predict(X)
        if self._inverse_transform is not None:
            pred_12m = self._inverse_transform(np.clip(pred_12m, 0, None))
            pred_36m = self._inverse_transform(np.clip(pred_36m, 0, None))
        else:
            pred_12m = np.clip(pred_12m, 0, None)
            pred_36m = np.clip(pred_36m, 0, None)

        # Enforce monotonic LTV horizons: 36m should be at least 12m.
        pred_36m = np.maximum(pred_36m, pred_12m)

        # Interpolate 24m as geometric mean of 12m and 36m
        pred_24m = np.sqrt(pred_12m * pred_36m)

        # Estimate "effective" weights by comparing to base models
        bgnbd_12m  = meta_features["bgnbd_ltv_12m"].to_numpy()
        trans_12m  = meta_features["transformer_ltv_12m"].to_numpy()

        denom = np.abs(pred_12m - bgnbd_12m) + np.abs(pred_12m - trans_12m) + 1e-6
        w_bgnbd = np.abs(pred_12m - trans_12m) / denom   # weight toward BG/NBD
        w_trans  = 1.0 - w_bgnbd

        result = pl.DataFrame({
            "customer_id":           meta_features["customer_id"].to_list(),
            "ltv_12m":               pred_12m.tolist(),
            "ltv_24m":               pred_24m.tolist(),
            "ltv_36m":               pred_36m.tolist(),
            "meta_weight_bgnbd":     np.clip(w_bgnbd, 0, 1).tolist(),
            "meta_weight_transformer": np.clip(w_trans, 0, 1).tolist(),
        })

        logger.info(
            "Fusion predictions: {} customers | mean LTV_36m=£{:.2f}",
            len(result), float(pred_36m.mean()),
        )
        return result

    def predict_single(
        self,
        meta_features_dict: dict[str, float],
    ) -> dict[str, float]:
        """
        Predict LTV for a single customer (real-time API scoring).

        Args:
            meta_features_dict: dict with feature_name → value

        Returns:
            dict with ltv_12m, ltv_24m, ltv_36m
        """
        model_12m, model_36m = self._require_models()

        x = np.array(
            [meta_features_dict.get(f, 0.0) for f in self._feature_names],
            dtype=np.float32,
        ).reshape(1, -1)

        ltv_12m = float(np.clip(model_12m.predict(x)[0], 0, None))
        ltv_36m = float(np.clip(model_36m.predict(x)[0], 0, None))
        ltv_36m = max(ltv_36m, ltv_12m)
        ltv_24m = float(np.sqrt(ltv_12m * ltv_36m))

        return {"ltv_12m": ltv_12m, "ltv_24m": ltv_24m, "ltv_36m": ltv_36m}

    # ── Validation ───────────────────────────────────────────

    def validate(
        self,
        meta_features: pl.DataFrame,
        targets: pl.DataFrame,
        bgnbd_baseline: pl.DataFrame | None = None,
        transformer_baseline: pl.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Evaluate fusion model on held-out data.

        Also computes improvement over BG/NBD-only and Transformer-only baselines.
        """
        from backend.ml.bgnbd_model import (
            compute_gini, compute_top_decile_lift, compute_calibration_error
        )

        model_12m, _model_36m = self._require_models()

        df = meta_features.join(
            targets.select(["customer_id", "actual_ltv_12m"]),
            on="customer_id",
            how="inner",
        )

        feature_cols = [c for c in self._feature_names if c in df.columns]
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y_true = df["actual_ltv_12m"].cast(pl.Float64).to_numpy()

        pred_12m = model_12m.predict(X)
        if self._inverse_transform is not None:
            pred_12m = self._inverse_transform(np.clip(pred_12m, 0, None))
        else:
            pred_12m = np.clip(pred_12m, 0, None)
        mean_ltv = float(y_true.mean()) if y_true.mean() > 0 else 1.0

        metrics = {
            "mae_ltv_12m":       float(np.mean(np.abs(y_true - pred_12m))),
            "mae_pct_12m":       float(np.mean(np.abs(y_true - pred_12m)) / mean_ltv),
            "rmse_ltv_12m":      float(np.sqrt(np.mean((y_true - pred_12m) ** 2))),
            "gini_coefficient":  compute_gini(y_true, pred_12m),
            "top_decile_lift":   compute_top_decile_lift(y_true, pred_12m),
            "calibration_error": compute_calibration_error(y_true, pred_12m),
            "n_customers_val":   len(y_true),
            "mean_actual_ltv":   mean_ltv,
        }

        # Compare to baselines
        if bgnbd_baseline is not None:
            bgnbd_joined = df.join(
                bgnbd_baseline.select(["customer_id",
                                       pl.col("ltv_12m").alias("bgnbd_ltv")]),
                on="customer_id", how="left"
            )
            bgnbd_pred = (
                bgnbd_joined["bgnbd_ltv"].fill_null(0).cast(pl.Float64).to_numpy()
            )
            bgnbd_mae = float(np.mean(np.abs(y_true - bgnbd_pred)))
            metrics["mae_bgnbd_12m"] = bgnbd_mae
            metrics["improvement_over_bgnbd_pct"] = float(
                (bgnbd_mae - metrics["mae_ltv_12m"]) / max(bgnbd_mae, 1e-9) * 100
            )

        if transformer_baseline is not None:
            trans_joined = df.join(
                transformer_baseline.select(["customer_id",
                                             pl.col("ltv_12m").alias("trans_ltv")]),
                on="customer_id", how="left"
            )
            trans_pred = (
                trans_joined["trans_ltv"].fill_null(0).cast(pl.Float64).to_numpy()
            )
            trans_mae = float(np.mean(np.abs(y_true - trans_pred)))
            metrics["mae_transformer_12m"] = trans_mae
            metrics["improvement_over_transformer_pct"] = float(
                (trans_mae - metrics["mae_ltv_12m"]) / max(trans_mae, 1e-9) * 100
            )

        logger.info("=== Fusion Validation Metrics ===")
        logger.info("  MAE LTV 12m:         £{:.2f}  ({:.1f}% of mean)",
                    metrics["mae_ltv_12m"], 100 * metrics["mae_pct_12m"])
        logger.info("  RMSE LTV 12m:        £{:.2f}", metrics["rmse_ltv_12m"])
        logger.info("  Gini:                {:.4f}  (target > 0.65)", metrics["gini_coefficient"])
        logger.info("  Top decile lift:     {:.2f}×  (target > 3.0×)", metrics["top_decile_lift"])
        logger.info("  Calibration error:   {:.4f}  (target < 0.10)", metrics["calibration_error"])

        if "improvement_over_bgnbd_pct" in metrics:
            logger.info("  vs BG/NBD:           {:.1f}% improvement",
                        metrics["improvement_over_bgnbd_pct"])
        if "improvement_over_transformer_pct" in metrics:
            logger.info("  vs Transformer:      {:.1f}% improvement",
                        metrics["improvement_over_transformer_pct"])

        self._train_metrics.update(metrics)
        return metrics

    # ── SHAP Explainability ───────────────────────────────────

    def compute_shap_values(
        self,
        meta_features: pl.DataFrame,
        max_samples: int = 500,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute SHAP values for the 12m LTV model.

        Returns:
            (shap_values ndarray (n, n_features), feature_names list)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap is required. Run: pip install shap")
        model_12m, _model_36m = self._require_models()

        feature_cols = [c for c in self._feature_names if c in meta_features.columns]
        sample = meta_features.sample(min(max_samples, len(meta_features)), seed=42)
        X = sample.select(feature_cols).to_numpy().astype(np.float32)

        explainer   = shap.TreeExplainer(model_12m)
        shap_values = explainer.shap_values(X)

        logger.info(
            "SHAP values computed — shape {} for {} features",
            shap_values.shape, len(feature_cols),
        )
        return shap_values, feature_cols

    def get_global_feature_importance(
        self,
        meta_features: pl.DataFrame,
        max_samples: int = 500,
    ) -> pl.DataFrame:
        """
        Global SHAP feature importance (mean |SHAP|) per feature.
        """
        shap_vals, feat_names = self.compute_shap_values(meta_features, max_samples)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        result = pl.DataFrame({
            "feature_name":  feat_names,
            "mean_abs_shap": mean_abs_shap.tolist(),
        }).sort("mean_abs_shap", descending=True)

        result = result.with_columns(
            pl.Series("rank", list(range(1, len(result) + 1)))
        )
        return result

    def get_customer_shap_explanation(
        self,
        customer_meta: dict[str, float],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Get per-customer SHAP-based explanation for the LTV prediction.

        Returns list of {feature, value, shap_contribution, description} dicts.
        """
        if not SHAP_AVAILABLE:
            return []
        model_12m, _model_36m = self._require_models()

        x = np.array(
            [customer_meta.get(f, 0.0) for f in self._feature_names],
            dtype=np.float32,
        ).reshape(1, -1)

        explainer = shap.TreeExplainer(model_12m)
        shap_vals = explainer.shap_values(x)[0]

        contributions = list(zip(self._feature_names, shap_vals))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)

        return [
            {
                "feature":            feat,
                "value":              float(customer_meta.get(feat, 0.0)),
                "shap_contribution":  float(shap_val),
                "direction":          "increases" if shap_val > 0 else "decreases",
            }
            for feat, shap_val in contributions[:top_n]
        ]

    # ── Serialisation ─────────────────────────────────────────

    def save_to_disk(self, path: str | Path) -> None:
        """Save both XGBoost models + metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_12m, model_36m = self._require_models()
        model_12m.save_model(str(path / f"{self.model_version}_12m.ubj"))
        model_36m.save_model(str(path / f"{self.model_version}_36m.ubj"))

        import pickle
        with open(path / f"{self.model_version}_meta.pkl", "wb") as f:
            pickle.dump({
                "model_version":   self.model_version,
                "xgb_params":      self.xgb_params,
                "feature_names":   self._feature_names,
                "train_metrics":   self._train_metrics,
                "target_transform": self._transform_name,
            }, f)
        logger.info("Fusion model saved → {}", path)

    @classmethod
    def load_from_disk(cls, path: str | Path, model_version: str) -> "XGBoostMetaLearner":
        """Load fusion model from disk."""
        import pickle
        path = Path(path)

        with open(path / f"{model_version}_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        instance = cls(model_version=model_version, xgb_params=meta["xgb_params"])
        instance._feature_names = meta["feature_names"]
        instance._train_metrics = meta["train_metrics"]
        if meta.get("target_transform") == "log1p":
            instance._target_transform = np.log1p
            instance._inverse_transform = np.expm1
            instance._transform_name = "log1p"

        instance.model_12m = xgb.XGBRegressor(**meta["xgb_params"])
        instance.model_12m.load_model(str(path / f"{model_version}_12m.ubj"))

        instance.model_36m = xgb.XGBRegressor(**meta["xgb_params"])
        instance.model_36m.load_model(str(path / f"{model_version}_36m.ubj"))

        instance._is_fitted = True
        logger.info("Fusion model loaded — {}", model_version)
        return instance

    # ── Persistence ──────────────────────────────────────────

    def save_registry(
        self,
        db_client: Any,
        bgnbd_version: str = "",
        transformer_version: str = "",
        causal_version: str = "",
        val_metrics: dict | None = None,
        wandb_run_id: str | None = None,
        pipeline_run_id: str | None = None,
        optuna_study_name: str | None = None,
        n_trials: int | None = None,
        best_params: dict | None = None,
    ) -> None:
        """Persist model registry entry to Supabase."""
        m = val_metrics or self._train_metrics
        improvement_bgnbd = m.get("improvement_over_bgnbd_pct")
        improvement_transformer = m.get("improvement_over_transformer_pct")
        if improvement_bgnbd is None:
            mae_base = m.get("mae_bgnbd_12m")
            mae_fusion = m.get("mae_ltv_12m")
            if mae_base not in (None, 0) and mae_fusion is not None:
                improvement_bgnbd = float(
                    (float(mae_base) - float(mae_fusion)) / max(float(mae_base), 1e-9) * 100
                )
        if improvement_transformer is None:
            mae_base = m.get("mae_transformer_12m")
            mae_fusion = m.get("mae_ltv_12m")
            if mae_base not in (None, 0) and mae_fusion is not None:
                improvement_transformer = float(
                    (float(mae_base) - float(mae_fusion)) / max(float(mae_base), 1e-9) * 100
                )

        record = {
            "model_version":            self.model_version,
            "trained_at":               datetime.now(timezone.utc).isoformat(),
            "bgnbd_model_version":      bgnbd_version,
            "transformer_model_version": transformer_version,
            "causal_model_version":     causal_version,
            "xgb_n_estimators":         self.xgb_params.get("n_estimators"),
            "xgb_max_depth":            self.xgb_params.get("max_depth"),
            "xgb_learning_rate":        self.xgb_params.get("learning_rate"),
            "xgb_subsample":            self.xgb_params.get("subsample"),
            "xgb_colsample_bytree":     self.xgb_params.get("colsample_bytree"),
            "xgb_min_child_weight":     self.xgb_params.get("min_child_weight"),
            "mae_ltv_12m":              m.get("mae_ltv_12m"),
            "mae_ltv_36m":              m.get("mae_ltv_36m"),
            "mae_pct_12m":              m.get("mae_pct_12m"),
            "rmse_ltv_12m":             m.get("rmse_ltv_12m"),
            "gini_coefficient":         m.get("gini_coefficient"),
            "top_decile_lift":          m.get("top_decile_lift"),
            "calibration_error":        m.get("calibration_error"),
            "improvement_over_bgnbd_pct": improvement_bgnbd,
            "improvement_over_transformer_pct": improvement_transformer,
            "n_customers_train":        m.get("n_train"),
            "n_customers_val":          m.get("n_customers_val"),
            "wandb_run_id":             wandb_run_id,
            "pipeline_run_id":          pipeline_run_id,
            "optuna_study_name":        optuna_study_name,
            "optuna_n_trials":          n_trials,
            "optuna_best_params":       json.dumps(best_params) if best_params else None,
            "shap_computed":            SHAP_AVAILABLE,
        }
        db_client.bulk_upsert(
            "fusion_model_registry",
            [record],
            conflict_columns=["model_version"],
        )
        logger.info("Fusion model registry saved — {}", self.model_version)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling this method.")

    def _require_models(self) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
        self._check_fitted()
        if self.model_12m is None or self.model_36m is None:
            raise RuntimeError("XGBoost models are not initialized.")
        return self.model_12m, self.model_36m
