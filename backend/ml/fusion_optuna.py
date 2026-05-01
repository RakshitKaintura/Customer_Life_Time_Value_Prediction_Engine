"""
Optuna hyperparameter tuning for the XGBoost Meta-Learner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger


def tune_fusion_optuna(
    meta_features_train: pl.DataFrame,
    targets_train: pl.DataFrame,
    meta_features_val: pl.DataFrame,
    targets_val: pl.DataFrame,
    n_trials: int = 30,
    study_name: str = "fusion_xgb_tuning",
    db_url: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """
    Optuna study to tune the XGBoost meta-learner.

    Searches over:
        n_estimators, max_depth, learning_rate,
        subsample, colsample_bytree, min_child_weight,
        reg_alpha, reg_lambda

    Objective: minimise MAE on validation set.

    Returns:
        (best_params dict, study object)
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ImportError:
        raise ImportError("optuna is required. Run: pip install optuna")

    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required.")

    from backend.ml.fusion import XGBoostMetaLearner, META_FEATURES

    # Pre-compute val targets once
    val_df = meta_features_val.join(
        targets_val.select(["customer_id", "actual_ltv_12m"]),
        on="customer_id", how="inner"
    )
    feature_cols = [c for c in META_FEATURES if c in meta_features_val.columns]
    X_val = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_df["actual_ltv_12m"].to_numpy()

    # Pre-compute train
    train_df = meta_features_train.join(
        targets_train.select(["customer_id", "actual_ltv_12m"]),
        on="customer_id", how="inner"
    )
    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_df["actual_ltv_12m"].to_numpy()

    def objective(trial: Any) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",    50, 500),
            "max_depth":        trial.suggest_int("max_depth",        2, 8),
            "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample",      0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha",      1e-5, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda",     1e-5, 10.0, log=True),
            "random_state":     42,
            "n_jobs":           -1,
            "objective":        "reg:squarederror",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred  = np.clip(model.predict(X_val), 0, None)
        mae   = float(np.mean(np.abs(y_val - pred)))
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = db_url or f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1, "objective": "reg:squarederror"})

    logger.info("Optuna best MAE: {:.4f}", study.best_value)
    logger.info("Optuna best params: {}", best_params)

    return best_params, study