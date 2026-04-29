"""
Causal ML Feature Attribution — EconML Double ML (DML) + CausalForest.

Goal: Identify which features CAUSE high LTV (not just correlate).
      Gives marketing and product teams actionable levers.

Pipeline:
  1. Define causal DAG for the customer journey
  2. Run Double ML for each potential treatment variable
  3. Estimate Conditional Average Treatment Effect (CATE) per customer
  4. Build firmographic LTV lookup table from CATE estimates (cold-start)

Theory (Double ML, Chernozhukov et al. 2018):
  Y = θ·T + g(X) + ε
  T = f(X) + η

  Step 1: Residualise Y and T on controls X using cross-fitting
  Step 2: Regress Y-residual on T-residual to get θ (causal effect)

  This removes the confounding effect of X on both Y and T,
  giving an unbiased estimate of the causal effect θ of T on Y.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DB_NUMERIC_14_4_ABS_MAX = 9_999_999_999.0  # NUMERIC(14,4) absolute upper bound is < 1e10
DB_NUMERIC_12_4_ABS_MAX = 99_999_999.0  # NUMERIC(12,4) absolute upper bound is < 1e8
DB_NUMERIC_8_6_ABS_MAX = 99.0  # NUMERIC(8,6) absolute upper bound is < 1e2
EXP_INPUT_CLIP = 20.0  # exp(20) ~= 4.85e8, keeps transformed effects in sane range


def _safe_expm1(values: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable expm1 with clipping to avoid overflow."""
    clipped = np.clip(values, -EXP_INPUT_CLIP, EXP_INPUT_CLIP)
    out = np.expm1(clipped)
    return np.nan_to_num(out, nan=0.0, posinf=DB_NUMERIC_14_4_ABS_MAX, neginf=-DB_NUMERIC_14_4_ABS_MAX)


def _clip_db_numeric(value: float | int | None, abs_max: float = DB_NUMERIC_14_4_ABS_MAX) -> float | None:
    """Clip a scalar to DB-safe numeric range and replace non-finite values."""
    if value is None:
        return None
    v = float(value)
    if not np.isfinite(v):
        return 0.0
    return float(np.clip(v, -abs_max, abs_max))


def _sanitize_effect_record(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure numeric effect fields are finite and fit target DB precision."""
    out = dict(row)
    numeric_limits = {
        "ate": DB_NUMERIC_14_4_ABS_MAX,
        "ate_lower_ci": DB_NUMERIC_14_4_ABS_MAX,
        "ate_upper_ci": DB_NUMERIC_14_4_ABS_MAX,
        "ate_stderr": DB_NUMERIC_14_4_ABS_MAX,
        "ate_pvalue": DB_NUMERIC_8_6_ABS_MAX,
        "cate_mean": DB_NUMERIC_14_4_ABS_MAX,
        "cate_std": DB_NUMERIC_12_4_ABS_MAX,
        "cate_min": DB_NUMERIC_14_4_ABS_MAX,
        "cate_max": DB_NUMERIC_14_4_ABS_MAX,
    }
    for field, abs_max in numeric_limits.items():
        if field in out:
            out[field] = _clip_db_numeric(out[field], abs_max=abs_max)
    if out.get("ate_pvalue") is not None:
        out["ate_pvalue"] = float(np.clip(out["ate_pvalue"], 0.0, 1.0))
    return out

try:
    from econml.dml import LinearDML, CausalForestDML
    from econml.inference import BootstrapInference
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logger.warning("econml not installed — causal models unavailable. Run: pip install econml")


# ─────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────

# Treatments: features whose causal effect on LTV we want to estimate
TREATMENT_DEFINITIONS = {
    "onboarding_completed": {
        "type":        "binary",
        "description": "Customer completed onboarding in first 7 days",
        "proxy_col":   "days_to_second_purchase",   # proxy: fast second purchase ≈ onboarding
        "proxy_fn":    lambda x: (x <= 7).astype(float),
    },
    "high_value_first_purchase": {
        "type":        "binary",
        "description": "First purchase was in top 40% of order values",
        "proxy_col":   "first_purchase_amount",
        "proxy_fn":    lambda x: (x >= np.percentile(x.dropna(), 60)).astype(float),
    },
    "multi_category_buyer": {
        "type":        "binary",
        "description": "Customer purchased from 3+ product categories",
        "proxy_col":   "unique_categories",
        "proxy_fn":    lambda x: (x >= 3).astype(float),
    },
    "fast_repeat_buyer": {
        "type":        "binary",
        "description": "Made second purchase within 30 days of first",
        "proxy_col":   "days_to_second_purchase",
        "proxy_fn":    lambda x: (x <= 30).astype(float),
    },
    "high_frequency": {
        "type":        "binary",
        "description": "Made 5+ purchases in observation window",
        "proxy_col":   "frequency",
        "proxy_fn":    lambda x: (x >= 5).astype(float),
    },
    "international_buyer": {
        "type":        "binary",
        "description": "Purchased from non-UK country",
        "proxy_col":   "multi_country",
        "proxy_fn":    lambda x: x.astype(float),
    },
}

# Control features (confounders): included in X to remove their influence
CONTROL_FEATURES = [
    "frequency",
    "recency_days",
    "t_days",
    "monetary_avg",
    "monetary_std",
    "orders_count",
    "avg_days_between_orders",
    "unique_products",
    "unique_categories",
]

# Outcome variable
OUTCOME_VARIABLE = "actual_ltv_12m"


# ─────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────

def prepare_causal_dataset(
    rfm_df: pl.DataFrame,
    outcome_col: str = OUTCOME_VARIABLE,
    min_ltv: float = 0.0,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Prepare the RFM DataFrame for causal inference.

    Returns:
        (pandas_df, available_treatments, available_controls)
    """
    # Convert to pandas for sklearn/econml
    df = rfm_df.to_pandas()

    # Filter: require valid outcome and key features
    df = df.dropna(subset=[outcome_col, "frequency", "monetary_avg", "t_days"])
    df = df[df[outcome_col] >= min_ltv].copy()

    # Fill missing values
    df["days_to_second_purchase"]   = df["days_to_second_purchase"].fillna(365)
    df["monetary_std"]              = df["monetary_std"].fillna(0)
    df["avg_days_between_orders"]   = df["avg_days_between_orders"].fillna(df["t_days"])
    df["unique_products"]           = df["unique_products"].fillna(1)
    df["unique_categories"]         = df["unique_categories"].fillna(1)
    df["multi_country"]             = df["multi_country"].fillna(False).astype(float)
    df["first_purchase_amount"]     = df["first_purchase_amount"].fillna(df["monetary_avg"])

    # Build treatment columns
    available_treatments = []
    for name, defn in TREATMENT_DEFINITIONS.items():
        col = defn["proxy_col"]
        if col in df.columns:
            fn  = defn["proxy_fn"]
            df[f"t_{name}"] = fn(df[col].fillna(0))
            available_treatments.append(name)

    # Available controls
    available_controls = [c for c in CONTROL_FEATURES if c in df.columns]

    # Log1p outcome (stabilises regression on skewed LTV)
    df["log_ltv"] = np.log1p(df[outcome_col])

    logger.info(
        "Causal dataset: {} customers, {} treatments, {} controls",
        len(df), len(available_treatments), len(available_controls),
    )
    return df, available_treatments, available_controls


# ─────────────────────────────────────────────────────────────
# Double ML estimator
# ─────────────────────────────────────────────────────────────

class DoubleMLEstimator:
    """
    EconML LinearDML estimator for a single treatment variable.

    Uses:
      - model_y: GradientBoostingRegressor (nuisance for outcome)
      - model_t: RandomForestClassifier    (nuisance for treatment)
      - Final stage: Linear regression with cross-fitting (k=5 folds)

    For continuous treatments:
      - model_t: GradientBoostingRegressor
    """

    def __init__(
        self,
        treatment_name: str,
        treatment_type: str = "binary",
        cv_folds: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        if not ECONML_AVAILABLE:
            raise ImportError("econml is required. Run: pip install econml dowhy")

        self.treatment_name = treatment_name
        self.treatment_type = treatment_type
        self.cv_folds       = cv_folds
        self.random_state   = random_state

        # Nuisance models
        self.model_y = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=random_state,
        )
        if treatment_type == "binary":
            self.model_t = RandomForestClassifier(
                n_estimators=100, max_depth=5,
                random_state=random_state, n_jobs=n_jobs,
            )
        else:
            self.model_t = GradientBoostingRegressor(
                n_estimators=100, max_depth=4,
                random_state=random_state,
            )

        self.estimator: LinearDML | None = None
        self._is_fitted = False
        self._ate: float = 0.0
        self._ate_stderr: float = 0.0

    def fit(
        self,
        df: pd.DataFrame,
        controls: list[str],
        outcome_col: str = "log_ltv",
    ) -> "DoubleMLEstimator":
        """Fit the Double ML estimator."""
        T_col = f"t_{self.treatment_name}"
        if T_col not in df.columns:
            raise ValueError(f"Treatment column '{T_col}' not found in dataframe")

        Y = df[outcome_col].values
        T = df[T_col].values
        X = df[controls].values

        # Scale controls for linear final stage
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        logger.info(
            "Fitting LinearDML for treatment='{}' (n={}, treatment_mean={:.3f})",
            self.treatment_name, len(Y), T.mean(),
        )

        self.estimator = LinearDML(
            model_y=self.model_y,
            model_t=self.model_t,
            cv=self.cv_folds,
            random_state=self.random_state,
            linear_first_stages=False,
            discrete_treatment=(self.treatment_type == "binary"),
        )

        self.estimator.fit(Y, T, X=X_scaled)
        self._is_fitted = True

        # Extract ATE
        try:
            ate_result = self.estimator.ate(X_scaled)
            self._ate    = float(_safe_expm1(ate_result))  # reverse log1p safely
            ate_inf      = self.estimator.ate_inference(X_scaled)
            self._ate_stderr = float(ate_inf.stderr_mean)
        except Exception as e:
            logger.warning("ATE extraction failed for {}: {}", self.treatment_name, e)
            self._ate = 0.0
            self._ate_stderr = 0.0

        logger.info(
            "  ATE({}) = {:.2f}  stderr = {:.2f}",
            self.treatment_name, self._ate, self._ate_stderr,
        )
        return self

    def estimate_cate(
        self,
        df: pd.DataFrame,
        controls: list[str],
    ) -> np.ndarray:
        """
        Estimate Conditional Average Treatment Effect (CATE) per customer.

        Returns np.ndarray of shape (n_customers,) with CATE values in £.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first")

        X = self._scaler.transform(df[controls].values)
        cate_log = self.estimator.effect(X)
        cate_ltv = _safe_expm1(cate_log.flatten())  # reverse log1p safely
        return cate_ltv

    def estimate_cate_with_ci(
        self,
        df: pd.DataFrame,
        controls: list[str],
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CATE with confidence intervals.

        Returns (cate, lower_ci, upper_ci)
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first")

        X = self._scaler.transform(df[controls].values)
        try:
            inf = self.estimator.effect_inference(X)
            cate_log = inf.point_estimate.flatten()
            lower_log = inf.conf_int(alpha=alpha)[0].flatten()
            upper_log = inf.conf_int(alpha=alpha)[1].flatten()
            return (
                _safe_expm1(cate_log),
                _safe_expm1(lower_log),
                _safe_expm1(upper_log),
            )
        except Exception as e:
            logger.warning("CI estimation failed for {}: {}", self.treatment_name, e)
            cate = self.estimate_cate(df, controls)
            return cate, cate * 0.7, cate * 1.3

    def get_ate_pvalue(self) -> float:
        """Approximate p-value from ATE / stderr ratio (z-test)."""
        if self._ate_stderr <= 0:
            return 1.0
        z = abs(self._ate / self._ate_stderr)
        from scipy import stats
        return float(2 * (1 - stats.norm.cdf(z)))

    @property
    def ate(self) -> float:
        return self._ate

    @property
    def ate_stderr(self) -> float:
        return self._ate_stderr


# ─────────────────────────────────────────────────────────────
# Causal Forest for heterogeneous effects
# ─────────────────────────────────────────────────────────────

class CausalForestEstimator:
    """
    EconML CausalForestDML for richer CATE estimation.

    Used when we want tree-based (non-linear) CATE estimates
    with honest confidence intervals via GRF.

    More expressive than LinearDML but slower to fit.
    """

    def __init__(
        self,
        treatment_name: str,
        n_estimators: int = 200,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ) -> None:
        if not ECONML_AVAILABLE:
            raise ImportError("econml is required")

        self.treatment_name  = treatment_name
        self.n_estimators    = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state    = random_state
        self.estimator: CausalForestDML | None = None
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        controls: list[str],
        outcome_col: str = "log_ltv",
    ) -> "CausalForestEstimator":
        T_col = f"t_{self.treatment_name}"
        Y = df[outcome_col].values
        T = df[T_col].values
        X = df[controls].values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        logger.info(
            "Fitting CausalForestDML for treatment='{}' (n_estimators={})",
            self.treatment_name, self.n_estimators,
        )

        self.estimator = CausalForestDML(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            discrete_treatment=(True),  # all our treatments are binary
            cv=3,
        )
        self.estimator.fit(Y, T, X=X_scaled)
        self._is_fitted = True

        logger.info("CausalForestDML fitted for '{}'", self.treatment_name)
        return self

    def estimate_cate(
        self, df: pd.DataFrame, controls: list[str]
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first")
        X = self._scaler.transform(df[controls].values)
        return _safe_expm1(self.estimator.effect(X).flatten())

    def estimate_cate_with_ci(
        self, df: pd.DataFrame, controls: list[str], alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first")
        X = self._scaler.transform(df[controls].values)
        inf = self.estimator.effect_inference(X)
        cate  = _safe_expm1(inf.point_estimate.flatten())
        lower = _safe_expm1(inf.conf_int(alpha=alpha)[0].flatten())
        upper = _safe_expm1(inf.conf_int(alpha=alpha)[1].flatten())
        return cate, lower, upper


# ─────────────────────────────────────────────────────────────
# Full Causal Pipeline
# ─────────────────────────────────────────────────────────────

class CausalLTVPipeline:
    """
    Orchestrates all causal ML steps.

    Usage:
        pipeline = CausalLTVPipeline(model_version='causal_v1')
        pipeline.fit(rfm_df)
        results = pipeline.get_results()
        pipeline.save(db_client)
    """

    def __init__(
        self,
        model_version: str = "causal_v1",
        use_causal_forest: bool = False,  # slow — use LinearDML by default
        outcome_col: str = OUTCOME_VARIABLE,
        cv_folds: int = 5,
    ) -> None:
        self.model_version   = model_version
        self.use_causal_forest = use_causal_forest
        self.outcome_col     = outcome_col
        self.cv_folds        = cv_folds

        self.estimators: dict[str, DoubleMLEstimator | CausalForestEstimator] = {}
        self.cate_results: dict[str, np.ndarray] = {}
        self.cate_ci_results: dict[str, tuple] = {}
        self._df: pd.DataFrame | None = None
        self._controls: list[str] = []
        self._treatments: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        rfm_df: pl.DataFrame,
    ) -> "CausalLTVPipeline":
        """
        Fit Double ML for each treatment variable.

        Args:
            rfm_df: Polars RFM DataFrame with actual_ltv_12m labels
        """
        df, treatments, controls = prepare_causal_dataset(
            rfm_df, outcome_col=self.outcome_col
        )
        self._df        = df
        self._controls  = controls
        self._treatments = treatments

        logger.info(
            "Fitting {} causal models (n_customers={}, n_controls={})",
            len(treatments), len(df), len(controls),
        )

        for treatment_name in treatments:
            defn = TREATMENT_DEFINITIONS[treatment_name]
            treatment_type = defn["type"]

            try:
                if self.use_causal_forest:
                    est = CausalForestEstimator(
                        treatment_name=treatment_name,
                        n_estimators=200,
                    )
                else:
                    est = DoubleMLEstimator(
                        treatment_name=treatment_name,
                        treatment_type=treatment_type,
                        cv_folds=self.cv_folds,
                    )

                est.fit(df, controls, outcome_col="log_ltv")
                self.estimators[treatment_name] = est

                # CATE per customer
                cate, lower, upper = est.estimate_cate_with_ci(df, controls)
                self.cate_results[treatment_name]    = cate
                self.cate_ci_results[treatment_name] = (cate, lower, upper)

            except Exception as exc:
                logger.error("Failed to fit treatment '{}': {}", treatment_name, exc)

        self._is_fitted = True
        logger.info(
            "Causal pipeline fitted — {} / {} treatments succeeded",
            len(self.estimators), len(treatments),
        )
        return self

    def get_treatment_effects_summary(self) -> pl.DataFrame:
        """
        Return a Polars DataFrame summarising ATE and significance for all treatments.
        """
        rows = []
        for name, est in self.estimators.items():
            defn = TREATMENT_DEFINITIONS[name]
            cate = self.cate_results[name]
            cate_ci = self.cate_ci_results[name]

            if isinstance(est, DoubleMLEstimator):
                ate        = est.ate
                ate_stderr = est.ate_stderr
                pvalue     = est.get_ate_pvalue()
                ate_lower  = ate - 1.96 * ate_stderr
                ate_upper  = ate + 1.96 * ate_stderr
            else:
                ate        = float(np.mean(cate))
                ate_stderr = float(np.std(cate) / np.sqrt(len(cate)))
                pvalue     = 0.05   # placeholder for CausalForest
                ate_lower  = float(np.mean(cate_ci[1]))
                ate_upper  = float(np.mean(cate_ci[2]))

            rows.append({
                "treatment_name":    name,
                "treatment_type":    defn["type"],
                "ate":               ate,
                "ate_lower_ci":      ate_lower,
                "ate_upper_ci":      ate_upper,
                "ate_stderr":        ate_stderr,
                "ate_pvalue":        pvalue,
                "effect_description": defn["description"],
                "effect_direction":  "positive" if ate > 0 else "negative",
                "is_significant":    pvalue < 0.05,
                "cate_mean":         float(np.mean(cate)),
                "cate_std":          float(np.std(cate)),
                "cate_min":          float(np.min(cate)),
                "cate_max":          float(np.max(cate)),
            })

        return pl.DataFrame(rows).sort("ate", descending=True)

    def get_customer_cate_df(self) -> pl.DataFrame:
        """
        Return a tall Polars DataFrame with per-customer CATE for each treatment.
        Columns: customer_id, treatment_name, cate_estimate, cate_lower, cate_upper
        """
        customer_ids = self._df["customer_id"].tolist()
        rows = []
        for name in self.estimators:
            cate, lower, upper = self.cate_ci_results[name]
            for cid, c, lo, hi in zip(customer_ids, cate, lower, upper):
                rows.append({
                    "customer_id":    cid,
                    "treatment_name": name,
                    "cate_estimate":  float(c),
                    "cate_lower":     float(lo),
                    "cate_upper":     float(hi),
                })
        return pl.DataFrame(rows)

    def get_top_lever_per_customer(self) -> pl.DataFrame:
        """
        For each customer, return the single treatment with the highest positive CATE.
        Used to populate causal_lever_recommendations table.
        """
        customer_ids = self._df["customer_id"].tolist()
        n = len(customer_ids)

        # Build matrix: (n_customers, n_treatments)
        treatment_names = list(self.cate_results.keys())
        cate_matrix = np.column_stack([
            self.cate_results[t] for t in treatment_names
        ])  # (n, k)

        best_idx    = np.argmax(cate_matrix, axis=1)
        best_effect = cate_matrix[np.arange(n), best_idx]

        rows = []
        for i, cid in enumerate(customer_ids):
            best_name = treatment_names[best_idx[i]]
            levers = [
                {
                    "lever":       t,
                    "effect":      float(cate_matrix[i, j]),
                    "description": TREATMENT_DEFINITIONS[t]["description"],
                }
                for j, t in enumerate(treatment_names)
                if cate_matrix[i, j] > 0
            ]
            levers.sort(key=lambda x: x["effect"], reverse=True)

            rows.append({
                "customer_id":           cid,
                "top_lever":             best_name,
                "top_lever_effect_usd":  float(best_effect[i]),
                "lever_json":            levers[:3],  # top 3 levers
            })

        return pl.DataFrame(rows)

    def save(
        self,
        db_client: Any,
        pipeline_run_id: str | None = None,
        wandb_run_id: str | None = None,
    ) -> None:
        """Persist all causal results to Supabase."""
        import json

        # 1. Model registry
        effects_df = self.get_treatment_effects_summary()
        estimator_type = "CausalForest" if self.use_causal_forest else "DML"
        # Normalize legacy names to match DB constraint values.
        estimator_type = {
            "LinearDML": "DML",
            "CausalForestDML": "CausalForest",
        }.get(estimator_type, estimator_type)
        db_client.bulk_upsert("causal_model_registry", [{
            "model_version":     self.model_version,
            "trained_at":        datetime.now(timezone.utc).isoformat(),
            "outcome_variable":  self.outcome_col,
            "estimator_type":    estimator_type,
            "n_treatments":      len(self.estimators),
            "n_controls":        len(self._controls),
            "n_customers":       len(self._df),
            "nuisance_cv_folds": self.cv_folds,
            "wandb_run_id":      wandb_run_id,
            "pipeline_run_id":   pipeline_run_id,
        }], conflict_columns=["model_version"])

        # 2. Treatment effects
        effects_records = effects_df.with_columns(
            pl.lit(self.model_version).alias("model_version"),
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("computed_at"),
        ).to_dicts()
        effects_records = [_sanitize_effect_record(r) for r in effects_records]
        db_client.bulk_upsert(
            "causal_treatment_effects",
            effects_records,
            conflict_columns=["model_version", "treatment_name"],
        )
        logger.info("Saved {} treatment effects", len(effects_records))

        # 3. Per-customer CATE
        cate_df = self.get_customer_cate_df()
        cate_records = cate_df.with_columns(
            pl.lit(self.model_version).alias("model_version"),
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("computed_at"),
        ).to_dicts()
        for r in cate_records:
            r["cate_estimate"] = _clip_db_numeric(r.get("cate_estimate"))
            r["cate_lower"] = _clip_db_numeric(r.get("cate_lower"))
            r["cate_upper"] = _clip_db_numeric(r.get("cate_upper"))
        db_client.bulk_upsert(
            "customer_cate",
            cate_records,
            conflict_columns=["customer_id", "model_version", "treatment_name"],
            batch_size=500,
        )
        logger.info("Saved {} customer CATE rows", len(cate_records))

        # 4. Top lever recommendations
        levers_df = self.get_top_lever_per_customer()
        levers_records = []
        for row in levers_df.iter_rows(named=True):
            levers_records.append({
                "customer_id":          row["customer_id"],
                "model_version":        self.model_version,
                "top_lever":            row["top_lever"],
                "top_lever_effect_usd": max(0.0, _clip_db_numeric(row["top_lever_effect_usd"]) or 0.0),
                "lever_json":           json.dumps(row["lever_json"]),
                "computed_at":          datetime.now(timezone.utc).isoformat(),
            })
        db_client.bulk_upsert(
            "causal_lever_recommendations",
            levers_records,
            conflict_columns=["customer_id", "model_version"],
            batch_size=500,
        )
        logger.info("Saved {} lever recommendations", len(levers_records))
