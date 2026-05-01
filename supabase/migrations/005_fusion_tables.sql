-- Migration 005 — Fusion Layer + Final LTV Scoring tables
BEGIN;

-- ============================================================
-- FUSION MODEL REGISTRY
-- ============================================================
CREATE TABLE IF NOT EXISTS fusion_model_registry (
    id                      BIGSERIAL       PRIMARY KEY,
    model_version           TEXT            NOT NULL UNIQUE,
    trained_at              TIMESTAMPTZ     DEFAULT NOW(),
    observation_end         DATE,

    -- Component model versions used
    bgnbd_model_version     TEXT,
    transformer_model_version TEXT,
    causal_model_version    TEXT,

    -- XGBoost meta-learner config
    xgb_n_estimators        INTEGER,
    xgb_max_depth           INTEGER,
    xgb_learning_rate       NUMERIC(8, 6),
    xgb_subsample           NUMERIC(6, 4),
    xgb_colsample_bytree    NUMERIC(6, 4),
    xgb_min_child_weight    INTEGER,

    -- Validation metrics
    mae_ltv_12m             NUMERIC(12, 4),
    mae_ltv_36m             NUMERIC(12, 4),
    mae_pct_12m             NUMERIC(8,  4),
    rmse_ltv_12m            NUMERIC(12, 4),
    gini_coefficient        NUMERIC(8,  6),
    top_decile_lift         NUMERIC(8,  4),
    calibration_error       NUMERIC(8,  6),

    -- vs individual models
    improvement_over_bgnbd_pct  NUMERIC(8, 4),
    improvement_over_transformer_pct NUMERIC(8, 4),

    -- SHAP
    shap_computed           BOOLEAN         DEFAULT FALSE,

    -- Tracking
    optuna_study_name       TEXT,
    optuna_n_trials         INTEGER,
    optuna_best_params      JSONB,
    wandb_run_id            TEXT,
    pipeline_run_id         TEXT,

    n_customers_train       INTEGER,
    n_customers_val         INTEGER
);

-- ============================================================
-- FINAL LTV SCORES  (replaces / extends ltv_predictions)
-- ============================================================
CREATE TABLE IF NOT EXISTS final_ltv_scores (
    id                          BIGSERIAL       PRIMARY KEY,
    customer_id                 TEXT            NOT NULL
                                REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_version               TEXT            NOT NULL,
    scored_at                   TIMESTAMPTZ     DEFAULT NOW(),
    ltv_source                  TEXT            NOT NULL DEFAULT 'full_model',

    -- Final ensemble predictions
    ltv_12m                     NUMERIC(14, 4),
    ltv_24m                     NUMERIC(14, 4),
    ltv_36m                     NUMERIC(14, 4),

    -- Component predictions
    bgnbd_ltv_12m               NUMERIC(14, 4),
    bgnbd_ltv_36m               NUMERIC(14, 4),
    transformer_ltv_12m         NUMERIC(14, 4),
    transformer_ltv_36m         NUMERIC(14, 4),

    -- Meta-learner weights learned for this customer
    meta_weight_bgnbd           NUMERIC(6, 4),
    meta_weight_transformer     NUMERIC(6, 4),

    -- BG/NBD auxiliary
    probability_alive_12m       NUMERIC(6,  4),
    expected_purchases_12m      NUMERIC(10, 4),

    -- Confidence intervals
    ci_lower_36m                NUMERIC(14, 4),
    ci_upper_36m                NUMERIC(14, 4),

    -- Segmentation
    ltv_percentile              SMALLINT,
    segment                     TEXT,
    recommended_max_cac         NUMERIC(12, 4),

    -- Explainability
    top_drivers                 JSONB,
    causal_levers               JSONB,
    shap_values                 JSONB,

    -- Scoring metadata
    scoring_latency_ms          INTEGER,
    inference_backend           TEXT            DEFAULT 'onnx',

    UNIQUE (customer_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_final_ltv_customer
    ON final_ltv_scores(customer_id);
CREATE INDEX IF NOT EXISTS idx_final_ltv_segment
    ON final_ltv_scores(segment);
CREATE INDEX IF NOT EXISTS idx_final_ltv_score
    ON final_ltv_scores(ltv_36m DESC);
CREATE INDEX IF NOT EXISTS idx_final_ltv_percentile
    ON final_ltv_scores(ltv_percentile);

-- ============================================================
-- SHAP FEATURE IMPORTANCE  (global + per-customer)
-- ============================================================
CREATE TABLE IF NOT EXISTS shap_global_importance (
    id              BIGSERIAL   PRIMARY KEY,
    model_version   TEXT        NOT NULL,
    feature_name    TEXT        NOT NULL,
    mean_abs_shap   NUMERIC(14, 6),
    rank            INTEGER,
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_version, feature_name)
);

-- ============================================================
-- LTV SEGMENT BOUNDARIES
-- ============================================================
CREATE TABLE IF NOT EXISTS ltv_segment_boundaries (
    id              BIGSERIAL   PRIMARY KEY,
    model_version   TEXT        NOT NULL UNIQUE,
    champions_min   NUMERIC(14, 4),
    high_value_min  NUMERIC(14, 4),
    medium_value_min NUMERIC(14, 4),
    low_value_min   NUMERIC(14, 4)   DEFAULT 0,
    p25             NUMERIC(14, 4),
    p50             NUMERIC(14, 4),
    p75             NUMERIC(14, 4),
    p90             NUMERIC(14, 4),
    p99             NUMERIC(14, 4),
    mean_ltv        NUMERIC(14, 4),
    computed_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- SCORING AUDIT LOG  (for latency monitoring)
-- ============================================================
CREATE TABLE IF NOT EXISTS scoring_audit (
    id                  BIGSERIAL   PRIMARY KEY,
    customer_id         TEXT,
    model_version       TEXT,
    ltv_source          TEXT,
    scoring_latency_ms  INTEGER,
    scored_at           TIMESTAMPTZ DEFAULT NOW(),
    api_endpoint        TEXT,
    error_message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_scored_at
    ON scoring_audit(scored_at DESC);

-- ============================================================
-- VIEWS
-- ============================================================

-- Full customer summary with final scores
CREATE OR REPLACE VIEW v_final_customer_scores AS
SELECT
    c.customer_id,
    c.country,
    c.acquisition_channel,
    c.vertical,
    c.plan_tier,
    c.first_purchase_date,
    c.total_orders,
    c.total_revenue,
    f.ltv_12m,
    f.ltv_24m,
    f.ltv_36m,
    f.segment,
    f.ltv_percentile,
    f.probability_alive_12m,
    f.recommended_max_cac,
    f.ci_lower_36m,
    f.ci_upper_36m,
    f.top_drivers,
    f.causal_levers,
    f.ltv_source,
    f.scored_at
FROM customers c
JOIN LATERAL (
    SELECT * FROM final_ltv_scores
    WHERE customer_id = c.customer_id
    ORDER BY scored_at DESC
    LIMIT 1
) f ON TRUE;

-- Segment distribution with revenue concentration
CREATE OR REPLACE VIEW v_segment_revenue_concentration AS
SELECT
    segment,
    COUNT(*)                                        AS n_customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_customers,
    SUM(ltv_36m)                                    AS total_ltv_36m,
    ROUND(100.0 * SUM(ltv_36m) / SUM(SUM(ltv_36m)) OVER(), 2) AS pct_revenue,
    AVG(ltv_36m)                                    AS avg_ltv_36m,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ltv_36m) AS median_ltv_36m,
    AVG(recommended_max_cac)                        AS avg_max_cac,
    AVG(probability_alive_12m)                      AS avg_p_alive
FROM final_ltv_scores
WHERE ltv_source = 'full_model'
GROUP BY segment
ORDER BY avg_ltv_36m DESC;

-- Model comparison view
CREATE OR REPLACE VIEW v_model_fusion_comparison AS
SELECT
    customer_id,
    bgnbd_ltv_36m,
    transformer_ltv_36m,
    ltv_36m AS fusion_ltv_36m,
    ltv_36m - bgnbd_ltv_36m         AS fusion_vs_bgnbd,
    ltv_36m - transformer_ltv_36m   AS fusion_vs_transformer,
    segment,
    ltv_percentile
FROM final_ltv_scores
WHERE ltv_source = 'full_model';

COMMIT;