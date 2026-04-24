-- Migration 002 - BG/NBD + Gamma-Gamma model outputs
-- Run after supabase/schema.sql and 001_initial_schema.sql

BEGIN;

-- ============================================================
-- BG/NBD MODEL PARAMETERS (one row per fitted model run)
-- ============================================================
CREATE TABLE IF NOT EXISTS bgnbd_model_params (
    id                      BIGSERIAL       PRIMARY KEY,
    model_version           TEXT            NOT NULL,
    fitted_at               TIMESTAMPTZ     DEFAULT NOW(),
    observation_end         DATE            NOT NULL,
    dataset                 TEXT            NOT NULL DEFAULT 'uci_online_retail',

    -- BG/NBD fitted parameters
    bgnbd_r                 NUMERIC(12, 8),
    bgnbd_alpha             NUMERIC(12, 8),
    bgnbd_a                 NUMERIC(12, 8),
    bgnbd_b                 NUMERIC(12, 8),

    -- Gamma-Gamma fitted parameters
    gg_p                    NUMERIC(12, 8),
    gg_q                    NUMERIC(12, 8),
    gg_v                    NUMERIC(12, 8),

    -- Fit quality
    bgnbd_log_likelihood    NUMERIC(16, 6),
    gg_log_likelihood       NUMERIC(16, 6),
    bgnbd_aic               NUMERIC(16, 6),
    gg_aic                  NUMERIC(16, 6),

    -- Holdout validation metrics
    mae_frequency_12m       NUMERIC(12, 6),
    rmse_frequency_12m      NUMERIC(12, 6),
    r2_frequency            NUMERIC(8,  6),
    mae_ltv_12m             NUMERIC(14, 4),
    mae_ltv_36m             NUMERIC(14, 4),
    gini_coefficient        NUMERIC(8,  6),
    top_decile_lift         NUMERIC(8,  4),
    calibration_error       NUMERIC(8,  6),

    -- Dataset split info
    n_customers_train       INTEGER,
    n_customers_holdout     INTEGER,
    observation_months      INTEGER,
    holdout_months          INTEGER,

    -- Tracking
    wandb_run_id            TEXT,
    pipeline_run_id         TEXT,

    UNIQUE (model_version),
    CHECK (bgnbd_r IS NULL OR bgnbd_r > 0),
    CHECK (bgnbd_alpha IS NULL OR bgnbd_alpha > 0),
    CHECK (bgnbd_a IS NULL OR bgnbd_a > 0),
    CHECK (bgnbd_b IS NULL OR bgnbd_b > 0),
    CHECK (gg_p IS NULL OR gg_p > 0),
    CHECK (gg_q IS NULL OR gg_q > 0),
    CHECK (gg_v IS NULL OR gg_v > 0),
    CHECK (r2_frequency IS NULL OR (r2_frequency >= -1 AND r2_frequency <= 1)),
    CHECK (gini_coefficient IS NULL OR (gini_coefficient >= 0 AND gini_coefficient <= 1)),
    CHECK (calibration_error IS NULL OR calibration_error >= 0),
    CHECK (top_decile_lift IS NULL OR top_decile_lift >= 0),
    CHECK (n_customers_train IS NULL OR n_customers_train >= 0),
    CHECK (n_customers_holdout IS NULL OR n_customers_holdout >= 0),
    CHECK (observation_months IS NULL OR observation_months >= 0),
    CHECK (holdout_months IS NULL OR holdout_months >= 0)
);

-- ============================================================
-- PER-CUSTOMER BG/NBD PREDICTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS bgnbd_predictions (
    id                          BIGSERIAL       PRIMARY KEY,
    customer_id                 TEXT            NOT NULL
                                REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_version               TEXT            NOT NULL
                                REFERENCES bgnbd_model_params(model_version) ON DELETE CASCADE,
    predicted_at                TIMESTAMPTZ     DEFAULT NOW(),

    -- BG/NBD outputs
    expected_purchases_90d      NUMERIC(10, 4),
    expected_purchases_180d     NUMERIC(10, 4),
    expected_purchases_365d     NUMERIC(10, 4),
    probability_alive           NUMERIC(6,  4),

    -- Gamma-Gamma outputs
    expected_avg_profit         NUMERIC(12, 4),

    -- LTV = E[X(t)] * E[avg_profit] * margin * discount
    ltv_12m                     NUMERIC(14, 4),
    ltv_24m                     NUMERIC(14, 4),
    ltv_36m                     NUMERIC(14, 4),

    -- Confidence intervals (Monte Carlo bootstrap)
    ltv_12m_lower               NUMERIC(14, 4),
    ltv_12m_upper               NUMERIC(14, 4),
    ltv_36m_lower               NUMERIC(14, 4),
    ltv_36m_upper               NUMERIC(14, 4),

    -- Holdout actuals (filled after holdout period)
    actual_purchases_holdout    INTEGER,
    actual_ltv_holdout          NUMERIC(14, 4),

    -- Segment (preliminary, before fusion)
    segment                     TEXT,

    UNIQUE (customer_id, model_version),
    CHECK (expected_purchases_90d IS NULL OR expected_purchases_90d >= 0),
    CHECK (expected_purchases_180d IS NULL OR expected_purchases_180d >= 0),
    CHECK (expected_purchases_365d IS NULL OR expected_purchases_365d >= 0),
    CHECK (probability_alive IS NULL OR (probability_alive >= 0 AND probability_alive <= 1)),
    CHECK (expected_avg_profit IS NULL OR expected_avg_profit >= 0),
    CHECK (ltv_12m IS NULL OR ltv_12m >= 0),
    CHECK (ltv_24m IS NULL OR ltv_24m >= 0),
    CHECK (ltv_36m IS NULL OR ltv_36m >= 0),
    CHECK (ltv_12m_lower IS NULL OR ltv_12m_lower >= 0),
    CHECK (ltv_12m_upper IS NULL OR ltv_12m_upper >= 0),
    CHECK (ltv_36m_lower IS NULL OR ltv_36m_lower >= 0),
    CHECK (ltv_36m_upper IS NULL OR ltv_36m_upper >= 0),
    CHECK (ltv_12m_lower IS NULL OR ltv_12m_upper IS NULL OR ltv_12m_lower <= ltv_12m_upper),
    CHECK (ltv_36m_lower IS NULL OR ltv_36m_upper IS NULL OR ltv_36m_lower <= ltv_36m_upper),
    CHECK (actual_purchases_holdout IS NULL OR actual_purchases_holdout >= 0),
    CHECK (actual_ltv_holdout IS NULL OR actual_ltv_holdout >= 0)
);

CREATE INDEX IF NOT EXISTS idx_bgnbd_pred_customer
    ON bgnbd_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_bgnbd_pred_model_version
    ON bgnbd_predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_bgnbd_pred_segment
    ON bgnbd_predictions(segment);
CREATE INDEX IF NOT EXISTS idx_bgnbd_pred_ltv
    ON bgnbd_predictions(ltv_36m DESC);

-- ============================================================
-- PROBABILITY-ALIVE MATRIX (for dashboard visualization)
-- ============================================================
CREATE TABLE IF NOT EXISTS probability_alive_matrix (
    id              BIGSERIAL   PRIMARY KEY,
    model_version   TEXT        NOT NULL
                    REFERENCES bgnbd_model_params(model_version) ON DELETE CASCADE,
    frequency       INTEGER     NOT NULL,
    recency_days    INTEGER     NOT NULL,
    t_days          INTEGER     NOT NULL DEFAULT 365,
    p_alive         NUMERIC(6, 4),
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_version, frequency, recency_days, t_days),
    CHECK (frequency >= 0),
    CHECK (recency_days >= 0),
    CHECK (t_days > 0),
    CHECK (recency_days <= t_days),
    CHECK (p_alive IS NULL OR (p_alive >= 0 AND p_alive <= 1))
);

-- ============================================================
-- CALIBRATION PLOT DATA (predicted vs actual for notebooks)
-- ============================================================
CREATE TABLE IF NOT EXISTS bgnbd_calibration (
    id                      BIGSERIAL   PRIMARY KEY,
    model_version           TEXT        NOT NULL
                            REFERENCES bgnbd_model_params(model_version) ON DELETE CASCADE,
    frequency_bucket        INTEGER,
    predicted_purchases_avg NUMERIC(10, 4),
    actual_purchases_avg    NUMERIC(10, 4),
    customer_count          INTEGER,
    computed_at             TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (model_version, frequency_bucket),
    CHECK (frequency_bucket IS NULL OR frequency_bucket >= 0),
    CHECK (predicted_purchases_avg IS NULL OR predicted_purchases_avg >= 0),
    CHECK (actual_purchases_avg IS NULL OR actual_purchases_avg >= 0),
    CHECK (customer_count IS NULL OR customer_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_prob_alive_model_version
    ON probability_alive_matrix(model_version);
CREATE INDEX IF NOT EXISTS idx_bgnbd_calibration_model_version
    ON bgnbd_calibration(model_version);

-- ============================================================
-- VIEWS
-- ============================================================

-- Latest BG/NBD predictions per customer
CREATE OR REPLACE VIEW v_latest_bgnbd AS
SELECT DISTINCT ON (customer_id)
    *
FROM bgnbd_predictions
ORDER BY customer_id, predicted_at DESC, model_version DESC;

-- BG/NBD vs actual performance by cohort
CREATE OR REPLACE VIEW v_bgnbd_cohort_performance AS
SELECT
    r.cohort_month,
    COUNT(*)                            AS customers,
    AVG(b.ltv_12m)                      AS avg_predicted_ltv_12m,
    AVG(b.ltv_36m)                      AS avg_predicted_ltv_36m,
    AVG(r.actual_ltv_12m)               AS avg_actual_ltv_12m,
    AVG(b.probability_alive)            AS avg_p_alive,
    AVG(ABS(b.ltv_12m - r.actual_ltv_12m))
                                        AS mae_12m
FROM v_latest_bgnbd b
JOIN v_latest_rfm r USING (customer_id)
WHERE r.actual_ltv_12m IS NOT NULL
GROUP BY r.cohort_month
ORDER BY r.cohort_month;

COMMIT;
