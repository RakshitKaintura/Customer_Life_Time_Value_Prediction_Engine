-- Migration 006 — MLOps monitoring, drift detection, retraining log
BEGIN;

-- ============================================================
-- LTV DRIFT ALERTS
-- ============================================================
CREATE TABLE IF NOT EXISTS ltv_drift_alerts (
    id                  BIGSERIAL       PRIMARY KEY,
    alert_id            TEXT            NOT NULL UNIQUE DEFAULT gen_random_uuid()::TEXT,
    detected_at         TIMESTAMPTZ     DEFAULT NOW(),
    alert_type          TEXT            NOT NULL,
    -- types: distribution_shift | mean_shift | segment_shift | feature_drift

    -- Drift metrics
    ks_statistic        NUMERIC(8, 6),   -- Kolmogorov-Smirnov test statistic
    ks_pvalue           NUMERIC(8, 6),
    psi_score           NUMERIC(8, 6),   -- Population Stability Index
    mean_shift_pct      NUMERIC(8, 4),   -- % change in mean LTV
    segment_shift_pct   NUMERIC(8, 4),   -- % change in segment distribution

    -- Thresholds
    threshold_exceeded  BOOLEAN         DEFAULT TRUE,
    threshold_value     NUMERIC(8, 6),
    actual_value        NUMERIC(8, 6),

    -- Context
    baseline_period_start   DATE,
    baseline_period_end     DATE,
    monitoring_period_start DATE,
    monitoring_period_end   DATE,

    n_baseline_customers    INTEGER,
    n_monitoring_customers  INTEGER,

    -- Status
    status              TEXT            DEFAULT 'open',
    -- open | acknowledged | resolved | false_positive
    resolved_at         TIMESTAMPTZ,
    notes               TEXT,

    model_version       TEXT,
    pipeline_run_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_drift_detected_at
    ON ltv_drift_alerts(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_status
    ON ltv_drift_alerts(status);

-- ============================================================
-- MODEL PERFORMANCE OVER TIME  (rolling validation)
-- ============================================================
CREATE TABLE IF NOT EXISTS model_performance_history (
    id                  BIGSERIAL       PRIMARY KEY,
    evaluated_at        TIMESTAMPTZ     DEFAULT NOW(),
    model_version       TEXT            NOT NULL,
    evaluation_type     TEXT            NOT NULL,
    -- rolling_validation | holdout | production

    -- Time window
    period_start        DATE,
    period_end          DATE,
    cohort_month        TEXT,

    -- Metrics
    mae_ltv_12m         NUMERIC(12, 4),
    rmse_ltv_12m        NUMERIC(12, 4),
    mae_pct             NUMERIC(8,  4),
    gini_coefficient    NUMERIC(8,  6),
    top_decile_lift     NUMERIC(8,  4),
    calibration_error   NUMERIC(8,  6),
    r2_score            NUMERIC(8,  6),

    n_customers         INTEGER,
    pipeline_run_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_perf_hist_model
    ON model_performance_history(model_version, evaluated_at DESC);

-- ============================================================
-- RETRAINING LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS retraining_log (
    id                  BIGSERIAL       PRIMARY KEY,
    run_id              TEXT            NOT NULL UNIQUE DEFAULT gen_random_uuid()::TEXT,
    triggered_at        TIMESTAMPTZ     DEFAULT NOW(),
    trigger_reason      TEXT            NOT NULL,
    -- scheduled | drift_alert | manual | performance_degradation

    -- Component versions
    prev_bgnbd_version  TEXT,
    new_bgnbd_version   TEXT,
    prev_transformer_version TEXT,
    new_transformer_version  TEXT,
    prev_fusion_version TEXT,
    new_fusion_version  TEXT,

    -- Outcomes
    status              TEXT            DEFAULT 'running',
    -- running | success | failed | rolled_back
    started_at          TIMESTAMPTZ     DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    duration_minutes    NUMERIC(8, 2),

    -- Metric comparison
    prev_gini           NUMERIC(8, 6),
    new_gini            NUMERIC(8, 6),
    prev_mae_pct        NUMERIC(8, 4),
    new_mae_pct         NUMERIC(8, 4),
    improvement_pct     NUMERIC(8, 4),

    deployed            BOOLEAN         DEFAULT FALSE,
    rollback_reason     TEXT,
    wandb_run_id        TEXT,
    notes               TEXT
);

-- ============================================================
-- FEATURE DRIFT TRACKING
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_drift_log (
    id                  BIGSERIAL       PRIMARY KEY,
    logged_at           TIMESTAMPTZ     DEFAULT NOW(),
    feature_name        TEXT            NOT NULL,
    model_version       TEXT,

    -- Distribution stats (baseline vs current)
    baseline_mean       NUMERIC(14, 6),
    baseline_std        NUMERIC(14, 6),
    current_mean        NUMERIC(14, 6),
    current_std         NUMERIC(14, 6),

    -- Drift measures
    psi_score           NUMERIC(8, 6),
    ks_statistic        NUMERIC(8, 6),
    mean_shift_pct      NUMERIC(10, 4),

    is_drifted          BOOLEAN         DEFAULT FALSE,
    drift_threshold     NUMERIC(8, 6)   DEFAULT 0.2,

    period_start        DATE,
    period_end          DATE
);

CREATE INDEX IF NOT EXISTS idx_feat_drift_feature
    ON feature_drift_log(feature_name, logged_at DESC);

-- ============================================================
-- ARIZE PHOENIX INTEGRATION LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS arize_sync_log (
    id              BIGSERIAL   PRIMARY KEY,
    synced_at       TIMESTAMPTZ DEFAULT NOW(),
    n_records       INTEGER,
    model_id        TEXT,
    model_version   TEXT,
    status          TEXT,
    error_message   TEXT
);

-- ============================================================
-- VIEWS
-- ============================================================

-- Latest drift status per alert type
CREATE OR REPLACE VIEW v_latest_drift_status AS
SELECT DISTINCT ON (alert_type)
    alert_id,
    alert_type,
    detected_at,
    psi_score,
    mean_shift_pct,
    ks_statistic,
    status,
    model_version
FROM ltv_drift_alerts
ORDER BY alert_type, detected_at DESC;

-- Retraining history summary
CREATE OR REPLACE VIEW v_retraining_summary AS
SELECT
    run_id,
    triggered_at,
    trigger_reason,
    status,
    prev_gini,
    new_gini,
    ROUND((new_gini - prev_gini) / NULLIF(prev_gini, 0) * 100, 2) AS gini_improvement_pct,
    duration_minutes,
    deployed
FROM retraining_log
ORDER BY triggered_at DESC;

-- Performance trend (last 30 days)
CREATE OR REPLACE VIEW v_performance_trend AS
SELECT
    evaluated_at::DATE          AS eval_date,
    model_version,
    evaluation_type,
    mae_ltv_12m,
    gini_coefficient,
    top_decile_lift,
    calibration_error,
    n_customers
FROM model_performance_history
WHERE evaluated_at > NOW() - INTERVAL '30 days'
ORDER BY evaluated_at DESC;

COMMIT;