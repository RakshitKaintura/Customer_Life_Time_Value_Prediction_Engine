-- Migration 003 - Transformer model outputs + ONNX tracking
BEGIN;

-- ============================================================
-- TRANSFORMER MODEL REGISTRY
-- ============================================================
CREATE TABLE IF NOT EXISTS transformer_model_registry (
    id                      BIGSERIAL       PRIMARY KEY,
    model_version           TEXT            NOT NULL UNIQUE,
    trained_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    observation_end         DATE,
    dataset                 TEXT            NOT NULL DEFAULT 'uci_online_retail',

    -- Architecture
    embedding_dim           INTEGER,
    n_layers                INTEGER,
    n_heads                 INTEGER,
    ffn_dim                 INTEGER,
    dropout                 NUMERIC(4, 3)   CHECK (dropout >= 0 AND dropout <= 1),
    max_sequence_length     INTEGER,
    vocab_size_category     INTEGER,
    vocab_size_channel      INTEGER,

    -- Training config
    learning_rate           NUMERIC(12, 8)  CHECK (learning_rate > 0),
    batch_size              INTEGER,
    epochs_trained          INTEGER,
    optimizer               TEXT,
    loss_function           TEXT,

    -- Training metrics
    train_loss_final        NUMERIC(12, 6),
    val_loss_final          NUMERIC(12, 6),
    best_val_loss           NUMERIC(12, 6),
    best_epoch              INTEGER,

    -- Holdout validation metrics
    mae_ltv_12m             NUMERIC(12, 4),
    mae_ltv_36m             NUMERIC(12, 4),
    mae_pct_12m             NUMERIC(8,  4),
    gini_coefficient        NUMERIC(8,  6),
    top_decile_lift         NUMERIC(8,  4),
    calibration_error       NUMERIC(8,  6),

    -- ONNX
    onnx_exported           BOOLEAN         NOT NULL DEFAULT FALSE,
    onnx_path               TEXT,
    onnx_inference_ms_avg   NUMERIC(8,  2),
    onnx_pytorch_mae_delta  NUMERIC(12, 8),  -- must be < 1e-5

    -- Optuna
    optuna_study_name       TEXT,
    optuna_n_trials         INTEGER,
    optuna_best_params      JSONB,

    -- Tracking
    wandb_run_id            TEXT,
    pipeline_run_id         TEXT,

    CHECK (embedding_dim IS NULL OR embedding_dim > 0),
    CHECK (n_layers IS NULL OR n_layers > 0),
    CHECK (n_heads IS NULL OR n_heads > 0),
    CHECK (ffn_dim IS NULL OR ffn_dim > 0),
    CHECK (max_sequence_length IS NULL OR max_sequence_length > 0),
    CHECK (vocab_size_category IS NULL OR vocab_size_category > 0),
    CHECK (vocab_size_channel IS NULL OR vocab_size_channel > 0),
    CHECK (batch_size IS NULL OR batch_size > 0),
    CHECK (epochs_trained IS NULL OR epochs_trained > 0),
    CHECK (best_epoch IS NULL OR best_epoch > 0),
    CHECK (onnx_inference_ms_avg IS NULL OR onnx_inference_ms_avg >= 0)
);

-- ============================================================
-- TRANSFORMER PER-CUSTOMER PREDICTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS transformer_predictions (
    id                      BIGSERIAL       PRIMARY KEY,
    customer_id             TEXT            NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_version           TEXT            NOT NULL
                            REFERENCES transformer_model_registry(model_version),
    predicted_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    ltv_12m                 NUMERIC(14, 4),
    ltv_24m                 NUMERIC(14, 4),
    ltv_36m                 NUMERIC(14, 4),

    -- Monte Carlo Dropout uncertainty
    ltv_12m_mean            NUMERIC(14, 4),
    ltv_12m_std             NUMERIC(12, 4),
    ltv_36m_mean            NUMERIC(14, 4),
    ltv_36m_std             NUMERIC(12, 4),
    ltv_12m_lower           NUMERIC(14, 4),
    ltv_12m_upper           NUMERIC(14, 4),
    ltv_36m_lower           NUMERIC(14, 4),
    ltv_36m_upper           NUMERIC(14, 4),

    -- Inference metadata
    inference_backend       TEXT            NOT NULL DEFAULT 'onnx',  -- onnx | pytorch
    inference_latency_ms    NUMERIC(8,  2),
    sequence_length         INTEGER,

    UNIQUE (customer_id, model_version),
    CHECK (ltv_12m IS NULL OR ltv_12m >= 0),
    CHECK (ltv_24m IS NULL OR ltv_24m >= 0),
    CHECK (ltv_36m IS NULL OR ltv_36m >= 0),
    CHECK (ltv_12m_std IS NULL OR ltv_12m_std >= 0),
    CHECK (ltv_36m_std IS NULL OR ltv_36m_std >= 0),
    CHECK (inference_latency_ms IS NULL OR inference_latency_ms >= 0),
    CHECK (sequence_length IS NULL OR sequence_length > 0),
    CHECK (inference_backend IN ('onnx', 'pytorch'))
);

CREATE INDEX IF NOT EXISTS idx_transformer_pred_customer
    ON transformer_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transformer_pred_ltv
    ON transformer_predictions(ltv_36m DESC);
CREATE INDEX IF NOT EXISTS idx_transformer_pred_predicted_at
    ON transformer_predictions(predicted_at DESC);

-- ============================================================
-- OPTUNA TRIAL LOG  (for dashboard / analysis)
-- ============================================================
CREATE TABLE IF NOT EXISTS optuna_trials (
    id              BIGSERIAL   PRIMARY KEY,
    study_name      TEXT        NOT NULL,
    trial_number    INTEGER     NOT NULL,
    params          JSONB,
    val_loss        NUMERIC(12, 6),
    mae_12m         NUMERIC(12, 4),
    gini            NUMERIC(8,  6),
    state           TEXT,       -- COMPLETE | PRUNED | FAIL
    duration_s      NUMERIC(10, 2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (study_name, trial_number)
);

-- ============================================================
-- VIEWS
-- ============================================================

-- Latest transformer predictions per customer
CREATE OR REPLACE VIEW v_latest_transformer AS
SELECT DISTINCT ON (customer_id)
    *
FROM transformer_predictions
ORDER BY customer_id, predicted_at DESC;

-- Side-by-side BG/NBD vs Transformer
CREATE OR REPLACE VIEW v_model_comparison AS
SELECT
    b.customer_id,
    b.ltv_12m                   AS bgnbd_ltv_12m,
    b.ltv_36m                   AS bgnbd_ltv_36m,
    b.probability_alive         AS bgnbd_p_alive,
    t.ltv_12m                   AS transformer_ltv_12m,
    t.ltv_36m                   AS transformer_ltv_36m,
    r.frequency,
    r.recency_days,
    r.monetary_avg,
    r.cohort_month
FROM v_latest_bgnbd b
JOIN v_latest_transformer t  USING (customer_id)
JOIN v_latest_rfm r          USING (customer_id);

COMMIT;
