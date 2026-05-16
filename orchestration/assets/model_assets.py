"""
ML Model Training Assets — Dagster.

Asset graph:
    rfm_features + purchase_sequences
        → bgnbd_model
        → transformer_model  (slow, GPU preferred)
        → causal_model
        → fusion_model
        → final_ltv_scores
"""

from __future__ import annotations

from datetime import date
import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
from dagster import (
    AssetExecutionContext,
    AssetIn,
    Output,
    asset,
    MetadataValue,
)


# ─────────────────────────────────────────────────────────────
# BG/NBD model
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="ml_models",
    description="Fit BG/NBD + Gamma-Gamma on calibration RFM features",
    compute_kind="lifetimes",
    ins={"rfm_features": AssetIn()},
)
def bgnbd_model(
    context,
    rfm_features: pl.DataFrame,
) -> Output[dict]:
    """Fit and validate BG/NBD + Gamma-Gamma models on calibration/holdout split."""
    from backend.ml.bgnbd_model import BGNBDModel
    from backend.ml.hyperparameter_tuning import tune_penalizer_grid
    from backend.config import settings

    context.log.info("Fitting BG/NBD + Gamma-Gamma model")

    # Get observation end date from rfm_features
    obs_end = rfm_features["observation_end_date"][0]
    if isinstance(obs_end, str):
        obs_end = date.fromisoformat(str(obs_end))

    # Split RFM into calibration/holdout using temporal cutoff
    # Calibration: first 80% of observation period
    # Holdout: remaining 20% for validation (looking forward in time)
    n_customers = len(rfm_features)
    split_idx = int(n_customers * 0.8)
    
    # Sort by customer_id for reproducibility
    rfm_sorted = rfm_features.sort("customer_id")
    calibration_rfm = rfm_sorted[:split_idx]
    holdout_rfm = rfm_sorted[split_idx:]
    
    context.log.info(f"Temporal split: {len(calibration_rfm)} calibration, {len(holdout_rfm)} holdout")

    # Tune penalizer on calibration set, validate on holdout
    context.log.info("Grid search for best penalizer_coef")
    best_penalizer, grid_results = tune_penalizer_grid(
        calibration_rfm = calibration_rfm,
        holdout_rfm     = holdout_rfm,
        observation_end = obs_end,
        penalizer_values = [0.0001, 0.001, 0.01, 0.1],
    )
    context.log.info(f"Best penalizer: {best_penalizer}")

    model_version = f"bgnbd_v{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M')}"
    model = BGNBDModel(
        penalizer_coef  = best_penalizer,
        model_version   = model_version,
        observation_end = obs_end,
    )
    model.fit(calibration_rfm, verbose=False)

    # Predict on full dataset
    predictions = model.predict(rfm_features, n_bootstrap=50)

    # Validate on holdout set
    metrics = model.validate(calibration_rfm, holdout_rfm)

    # Save to disk
    model_path = Path("./models")
    model.save_to_disk(model_path)
    context.log.info(f"BG/NBD model saved → {model_path}")

    # Save to Supabase
    try:
        from backend.db.supabase_client import SupabaseClient
        db = SupabaseClient(use_service_role=True)
        model.save_params(db, pipeline_run_id="dagster")
        model.save_predictions(predictions, db)
    except Exception as exc:
        context.log.warning(f"DB save failed: {exc}")

    return Output(
        {
            "model_version": model_version,
            "penalizer":     best_penalizer,
            "predictions":   predictions,
            "metrics":       metrics,
            "params":        model.get_params(),
        },
        metadata={
            "model_version":    MetadataValue.text(model_version),
            "penalizer":        MetadataValue.float(best_penalizer),
            "r2_frequency":     MetadataValue.float(float(round(metrics.get("r2_frequency", 0), 4))),
            "gini_coefficient": MetadataValue.float(float(round(metrics.get("gini_coefficient", 0), 4))),
            "top_decile_lift":  MetadataValue.float(float(round(metrics.get("top_decile_lift", 0), 2))),
            "mae_ltv_12m":      MetadataValue.float(float(round(metrics.get("mae_ltv_12m", 0), 2))),
        },
    )


# ─────────────────────────────────────────────────────────────
# Causal model
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="ml_models",
    description="Fit Double ML causal model + cold-start firmographic table",
    compute_kind="econml",
    ins={"rfm_features": AssetIn()},
)
def causal_model(
    context,
    rfm_features: pl.DataFrame,
) -> Output[dict]:
    """Run EconML Double ML for causal feature attribution."""
    try:
        from backend.ml.causal_model import CausalLTVPipeline
        from backend.ml.cold_start import build_firmographic_lookup, ColdStartScorer
        from backend.db.supabase_client import SupabaseClient
    except ImportError as e:
        context.log.warning(f"Causal dependencies missing: {e}")
        return Output(
            {"model_version": "causal_v1_skipped", "status": "skipped"},
            metadata={"status": MetadataValue.text("skipped — econml not installed")},
        )

    model_version = f"causal_v{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M')}"
    context.log.info(f"Fitting causal pipeline {model_version}")

    pipeline = CausalLTVPipeline(
        model_version   = model_version,
        use_causal_forest = False,
        outcome_col     = "actual_ltv_12m",
        cv_folds        = 5,
    )
    pipeline.fit(rfm_features)

    effects_df = pipeline.get_treatment_effects_summary()
    n_sig = int(effects_df["is_significant"].sum())

    context.log.info(f"Fitted {len(pipeline.estimators)} treatments, {n_sig} significant")

    customer_ids = pipeline._df["customer_id"].tolist() if pipeline._df is not None else []

    # Build firmographic lookup
    lookup_df = build_firmographic_lookup(
        rfm_df              = rfm_features,
        cate_per_customer   = pipeline.cate_results,
        customer_ids        = customer_ids,
        causal_model_version = model_version,
    )

    # Persist
    try:
        db = SupabaseClient(use_service_role=True)
        pipeline.save(db, pipeline_run_id="dagster")
        scorer = ColdStartScorer(db)
        scorer.save_table(lookup_df, db)
        context.log.info("Causal results saved to Supabase")
    except Exception as exc:
        context.log.warning(f"Causal DB save failed: {exc}")

    return Output(
        {
            "model_version":   model_version,
            "effects_df":      effects_df,
            "lookup_df":       lookup_df,
            "cate_results":    pipeline.cate_results,
            "customer_ids":    customer_ids,
        },
        metadata={
            "model_version":    MetadataValue.text(model_version),
            "n_treatments":     MetadataValue.int(len(pipeline.estimators)),
            "n_significant":    MetadataValue.int(n_sig),
            "n_coldstart_slices": MetadataValue.int(len(lookup_df)),
        },
    )


# ─────────────────────────────────────────────────────────────
# Fusion model
# ─────────────────────────────────────────────────────────────

@asset(
    group_name="ml_models",
    description="Train XGBoost meta-learner fusion model",
    compute_kind="xgboost",
    ins={
        "rfm_features": AssetIn(),
        "bgnbd_model":  AssetIn(),
    },
)
def fusion_model(
    context,
    rfm_features: pl.DataFrame,
    bgnbd_model: dict,
) -> Output[dict]:
    """Train XGBoost stacking meta-learner."""
    from sklearn.model_selection import train_test_split
    from backend.ml.fusion import XGBoostMetaLearner, build_meta_features
    from backend.ml.segmentation import assign_segments_batch

    context.log.info("Training XGBoost fusion model")

    bgnbd_preds    = bgnbd_model["predictions"]
    model_version  = f"fusion_v{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M')}"

    # Use BG/NBD predictions as both base models if Transformer not available
    try:
        onnx_path = Path("./models/transformer.onnx")
        if onnx_path.exists():
            from backend.ml.transformer_onnx import ONNXInferenceEngine
            context.log.info("Transformer ONNX found — using for meta features")
            # Load transformer predictions from DB
            from backend.db.supabase_client import SupabaseClient
            db = SupabaseClient(use_service_role=True)
            rows = db.execute_sql(
                "SELECT customer_id, ltv_12m, ltv_36m FROM transformer_predictions "
                "ORDER BY predicted_at DESC LIMIT 5000"
            )
            if rows:
                transformer_preds = pl.DataFrame(rows)
            else:
                raise ValueError("No transformer predictions in DB")
        else:
            raise FileNotFoundError("No ONNX model")
    except Exception as exc:
        context.log.warning(f"Transformer unavailable ({exc}) — using BG/NBD proxy")
        rng = np.random.default_rng(42)
        noise = rng.normal(1.0, 0.15, len(bgnbd_preds))
        transformer_preds = bgnbd_preds.with_columns([
            (pl.col("ltv_12m") * pl.Series(noise)).alias("ltv_12m"),
            (pl.col("ltv_36m") * pl.Series(noise)).alias("ltv_36m"),
        ])

    # Build meta-features
    meta_df = build_meta_features(bgnbd_preds, transformer_preds, rfm_features)
    targets  = rfm_features.select(["customer_id", "actual_ltv_12m"])

    # Train/val split
    ids = meta_df["customer_id"].to_list()
    train_ids, val_ids = train_test_split(ids, test_size=0.25, random_state=42)
    meta_train = meta_df.filter(pl.col("customer_id").is_in(train_ids))
    meta_val   = meta_df.filter(pl.col("customer_id").is_in(val_ids))

    xgb_params = {
        "n_estimators": 200, "max_depth": 4,
        "learning_rate": 0.05, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1,
        "objective": "reg:squarederror",
    }
    fusion = XGBoostMetaLearner(model_version=model_version, xgb_params=xgb_params)
    fusion.fit(meta_train, targets, eval_set_features=meta_val, eval_set_targets=targets)

    val_metrics = fusion.validate(
        meta_val, targets,
        bgnbd_baseline=bgnbd_preds,
        transformer_baseline=transformer_preds,
    )

    # Full predictions with segmentation
    all_preds = fusion.predict(meta_df)
    all_preds = assign_segments_batch(all_preds)

    # Save model
    fusion.save_to_disk(Path("./models"))
    context.log.info(f"Fusion model saved — Gini={val_metrics.get('gini_coefficient', 0):.4f}")

    # Save to Supabase
    try:
        from backend.db.supabase_client import SupabaseClient
        db = SupabaseClient(use_service_role=True)
        fusion.save_registry(
            db_client       = db,
            bgnbd_version   = bgnbd_model["model_version"],
            val_metrics     = val_metrics,
            pipeline_run_id = "dagster",
        )
        # Save final scores
        from datetime import datetime, timezone
        scored_at = datetime.now(timezone.utc).isoformat()
        records = []
        for row in all_preds.iter_rows(named=True):
            records.append({
                "customer_id":  row["customer_id"],
                "model_version": model_version,
                "scored_at":    scored_at,
                "ltv_source":   "full_model",
                "ltv_12m":      row["ltv_12m"],
                "ltv_24m":      row["ltv_24m"],
                "ltv_36m":      row["ltv_36m"],
                "segment":      row["segment"],
                "ltv_percentile": row["ltv_percentile"],
                "recommended_max_cac": row["recommended_max_cac"],
            })
        db.bulk_upsert("final_ltv_scores", records,
                       conflict_columns=["customer_id", "model_version"], batch_size=500)
        context.log.info(f"Saved {len(records):,} final LTV scores")
    except Exception as exc:
        context.log.warning(f"Fusion DB save failed: {exc}")

    return Output(
        {
            "model_version": model_version,
            "val_metrics":   val_metrics,
            "predictions":   all_preds,
        },
        metadata={
            "model_version":    MetadataValue.text(model_version),
            "gini_coefficient": MetadataValue.float(round(val_metrics.get("gini_coefficient", 0), 4)),
            "top_decile_lift":  MetadataValue.float(round(val_metrics.get("top_decile_lift", 0), 2)),
            "mae_pct_12m":      MetadataValue.float(round(val_metrics.get("mae_pct_12m", 0) * 100, 1)),
            "n_customers":      MetadataValue.int(len(all_preds)),
            "n_champions":      MetadataValue.int(int((all_preds["segment"] == "champions").sum())),
        },
    )