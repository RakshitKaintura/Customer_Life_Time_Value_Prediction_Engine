-- ============================================================
-- LTV Prediction Engine — Full Database Schema
-- Supabase PostgreSQL + pgvector
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- RAW TRANSACTIONS (UCI Online Retail Dataset, as-loaded)
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_transactions (
    id              BIGSERIAL PRIMARY KEY,
    invoice_no      TEXT        NOT NULL,
    stock_code      TEXT        NOT NULL,
    description     TEXT,
    quantity        INTEGER     NOT NULL,
    invoice_date    TIMESTAMPTZ NOT NULL,
    unit_price      NUMERIC(10, 4) NOT NULL,
    customer_id     TEXT,
    country         TEXT,
    source_dataset  TEXT        NOT NULL DEFAULT 'uci_online_retail',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_txn_customer   ON raw_transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_raw_txn_date       ON raw_transactions(invoice_date);
CREATE INDEX IF NOT EXISTS idx_raw_txn_invoice    ON raw_transactions(invoice_no);

-- ============================================================
-- CLEANED TRANSACTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS transactions (
    id                  BIGSERIAL PRIMARY KEY,
    invoice_no          TEXT           NOT NULL,
    stock_code          TEXT           NOT NULL,
    description         TEXT,
    quantity            INTEGER        NOT NULL,
    invoice_date        TIMESTAMPTZ    NOT NULL,
    unit_price          NUMERIC(10, 4) NOT NULL,
    line_total          NUMERIC(12, 4) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    customer_id         TEXT           NOT NULL,
    country             TEXT,
    product_category    TEXT,
    amount_bucket       SMALLINT,       -- 1–5 quantile bucket of line_total
    is_returned         BOOLEAN        DEFAULT FALSE,
    source_dataset      TEXT           NOT NULL DEFAULT 'uci_online_retail',
    created_at          TIMESTAMPTZ    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_txn_customer_id    ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_invoice_date   ON transactions(invoice_date);
CREATE INDEX IF NOT EXISTS idx_txn_customer_date  ON transactions(customer_id, invoice_date);
CREATE INDEX IF NOT EXISTS idx_txn_stock_code     ON transactions(stock_code);
CREATE INDEX IF NOT EXISTS idx_txn_category       ON transactions(product_category);

-- ============================================================
-- CUSTOMERS
-- ============================================================
CREATE TABLE IF NOT EXISTS customers (
    customer_id             TEXT           PRIMARY KEY,
    country                 TEXT,
    first_purchase_date     TIMESTAMPTZ,
    last_purchase_date      TIMESTAMPTZ,
    total_orders            INTEGER        DEFAULT 0,
    total_revenue           NUMERIC(14, 4) DEFAULT 0,
    acquisition_channel     TEXT           DEFAULT 'organic',
    vertical                TEXT,
    company_size            TEXT,
    plan_tier               TEXT,
    is_synthetic            BOOLEAN        DEFAULT FALSE,
    source_dataset          TEXT           DEFAULT 'uci_online_retail',
    created_at              TIMESTAMPTZ    DEFAULT NOW(),
    updated_at              TIMESTAMPTZ    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_customers_country     ON customers(country);
CREATE INDEX IF NOT EXISTS idx_customers_channel     ON customers(acquisition_channel);
CREATE INDEX IF NOT EXISTS idx_customers_first_date  ON customers(first_purchase_date);

-- ============================================================
-- RFM FEATURES  (computed by Polars pipeline)
-- ============================================================
CREATE TABLE IF NOT EXISTS rfm_features (
    id                          BIGSERIAL      PRIMARY KEY,
    customer_id                 TEXT           NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    computed_at                 TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    observation_end_date        DATE           NOT NULL,    -- last day of observation window

    -- Core BG/NBD inputs
    recency_days                NUMERIC(10, 2),   -- days from first to last purchase
    frequency                   INTEGER,           -- repeat purchase count (x in BG/NBD)
    monetary_avg                NUMERIC(12, 4),    -- avg order value (for Gamma-Gamma)
    t_days                      INTEGER,           -- customer age in days (T in BG/NBD)

    -- Extended monetary
    monetary_total              NUMERIC(14, 4),    -- total spend in observation window
    monetary_std                NUMERIC(12, 4),    -- std dev of order values
    purchase_variance           NUMERIC(14, 4),    -- variance of order values

    -- Purchase behaviour
    orders_count                INTEGER,
    unique_products             INTEGER,
    unique_categories           INTEGER,
    unique_invoices             INTEGER,
    avg_items_per_order         NUMERIC(8, 2),
    avg_days_between_orders     NUMERIC(10, 2),
    std_days_between_orders     NUMERIC(10, 2),

    -- Cohort / onboarding signals
    cohort_month                TEXT,              -- YYYY-MM of first purchase
    days_to_second_purchase     INTEGER,
    first_purchase_category     TEXT,
    first_purchase_amount       NUMERIC(12, 4),

    -- Derived flags
    has_returned_items          BOOLEAN        DEFAULT FALSE,
    multi_country               BOOLEAN        DEFAULT FALSE,

    -- Holdout ground-truth labels (filled after hold-out period elapses)
    actual_ltv_12m              NUMERIC(14, 4),
    actual_ltv_24m              NUMERIC(14, 4),
    actual_ltv_36m              NUMERIC(14, 4),

    -- Pipeline metadata
    pipeline_run_id             TEXT,

    UNIQUE (customer_id, observation_end_date)
);

CREATE INDEX IF NOT EXISTS idx_rfm_customer_id    ON rfm_features(customer_id);
CREATE INDEX IF NOT EXISTS idx_rfm_obs_end        ON rfm_features(observation_end_date);
CREATE INDEX IF NOT EXISTS idx_rfm_cohort         ON rfm_features(cohort_month);

-- ============================================================
-- PURCHASE SEQUENCES  (built in Phase 3 — schema ready now)
-- ============================================================
CREATE TABLE IF NOT EXISTS purchase_sequences (
    id              BIGSERIAL      PRIMARY KEY,
    customer_id     TEXT           NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    sequence_json   JSONB          NOT NULL,   -- [{cat_id, amount_bucket, days_delta, channel_id}, ...]
    sequence_length INTEGER        NOT NULL,
    observation_end_date DATE,
    built_at        TIMESTAMPTZ    DEFAULT NOW(),
    UNIQUE (customer_id)
);

-- ============================================================
-- CUSTOMER EMBEDDINGS  (pgvector — populated in Phase 3)
-- ============================================================
CREATE TABLE IF NOT EXISTS customer_embeddings (
    id              BIGSERIAL      PRIMARY KEY,
    customer_id     TEXT           NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    embedding       vector(64),                -- Transformer CLS token
    model_version   TEXT           NOT NULL,
    created_at      TIMESTAMPTZ    DEFAULT NOW(),
    UNIQUE (customer_id, model_version)
);

-- HNSW index for fast ANN lookalike search
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON customer_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================
-- LTV PREDICTIONS  (populated in Phase 5+)
-- ============================================================
CREATE TABLE IF NOT EXISTS ltv_predictions (
    id                      BIGSERIAL      PRIMARY KEY,
    customer_id             TEXT           NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    predicted_at            TIMESTAMPTZ    DEFAULT NOW(),
    model_version           TEXT           NOT NULL,
    ltv_source              TEXT           NOT NULL DEFAULT 'full_model',
                                           -- full_model | firmographic_prior | bgnbd_only

    -- Final ensemble predictions
    ltv_12m                 NUMERIC(14, 4),
    ltv_24m                 NUMERIC(14, 4),
    ltv_36m                 NUMERIC(14, 4),

    -- BG/NBD component
    bgnbd_ltv_12m           NUMERIC(14, 4),
    bgnbd_ltv_36m           NUMERIC(14, 4),
    probability_alive_12m   NUMERIC(6,  4),
    expected_purchases_12m  NUMERIC(10, 4),

    -- Transformer component
    transformer_ltv_12m     NUMERIC(14, 4),
    transformer_ltv_36m     NUMERIC(14, 4),

    -- Uncertainty
    ci_lower_36m            NUMERIC(14, 4),
    ci_upper_36m            NUMERIC(14, 4),

    -- Segmentation
    ltv_percentile          SMALLINT,
    segment                 TEXT,          -- champions | high_value | medium_value | low_value
    recommended_max_cac     NUMERIC(12, 4),

    -- Explainability
    top_drivers             JSONB,
    causal_levers           JSONB,
    shap_values             JSONB,

    -- Meta
    scoring_latency_ms      INTEGER,
    pipeline_run_id         TEXT,

    UNIQUE (customer_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_ltv_customer_id  ON ltv_predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_ltv_segment      ON ltv_predictions(segment);
CREATE INDEX IF NOT EXISTS idx_ltv_predicted_at ON ltv_predictions(predicted_at);

-- ============================================================
-- FIRMOGRAPHIC COLD-START LTV  (populated in Phase 4)
-- ============================================================
CREATE TABLE IF NOT EXISTS firmographic_ltv (
    id                  BIGSERIAL      PRIMARY KEY,
    vertical            TEXT           NOT NULL,
    company_size        TEXT           NOT NULL,
    channel             TEXT           NOT NULL,
    plan_tier           TEXT           NOT NULL,
    ltv_36m_estimate    NUMERIC(14, 4),
    ci_lower            NUMERIC(14, 4),
    ci_upper            NUMERIC(14, 4),
    cate_effect         NUMERIC(14, 4),
    sample_size         INTEGER,
    computed_at         TIMESTAMPTZ    DEFAULT NOW(),
    model_version       TEXT,
    UNIQUE (vertical, company_size, channel, plan_tier)
);

-- ============================================================
-- MODEL PERFORMANCE TRACKING
-- ============================================================
CREATE TABLE IF NOT EXISTS model_performance (
    id                  BIGSERIAL      PRIMARY KEY,
    model_name          TEXT           NOT NULL,
    model_version       TEXT           NOT NULL,
    horizon             TEXT           NOT NULL,  -- 12m | 24m | 36m
    cohort_month        TEXT,
    mae                 NUMERIC(12, 4),
    rmse                NUMERIC(12, 4),
    mape                NUMERIC(8,  4),
    gini                NUMERIC(6,  4),
    top_decile_lift     NUMERIC(6,  4),
    calibration_err     NUMERIC(8,  4),
    r2                  NUMERIC(6,  4),
    n_customers         INTEGER,
    evaluated_at        TIMESTAMPTZ    DEFAULT NOW()
);

-- ============================================================
-- PIPELINE RUN LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                  BIGSERIAL      PRIMARY KEY,
    run_id              TEXT           NOT NULL UNIQUE DEFAULT gen_random_uuid()::TEXT,
    pipeline_name       TEXT           NOT NULL,
    status              TEXT           NOT NULL DEFAULT 'running',
                        -- running | success | failed | cancelled
    started_at          TIMESTAMPTZ    DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    records_processed   INTEGER,
    error_message       TEXT,
    metadata            JSONB,
    wandb_run_id        TEXT
);

-- ============================================================
-- VIEWS — handy for the dashboard
-- ============================================================

-- Latest RFM per customer
CREATE OR REPLACE VIEW v_latest_rfm AS
SELECT DISTINCT ON (customer_id)
    *
FROM rfm_features
ORDER BY customer_id, observation_end_date DESC;

-- Customer summary with latest prediction
CREATE OR REPLACE VIEW v_customer_summary AS
SELECT
    c.customer_id,
    c.country,
    c.acquisition_channel,
    c.vertical,
    c.plan_tier,
    c.first_purchase_date,
    c.last_purchase_date,
    c.total_orders,
    c.total_revenue,
    r.frequency,
    r.recency_days,
    r.monetary_avg,
    r.monetary_total,
    r.actual_ltv_12m,
    r.cohort_month,
    p.ltv_12m,
    p.ltv_36m,
    p.segment,
    p.probability_alive_12m,
    p.recommended_max_cac,
    p.ltv_source,
    p.predicted_at
FROM customers c
LEFT JOIN v_latest_rfm r         ON r.customer_id = c.customer_id
LEFT JOIN LATERAL (
    SELECT * FROM ltv_predictions
    WHERE customer_id = c.customer_id
    ORDER BY predicted_at DESC
    LIMIT 1
) p ON TRUE;

-- Cohort LTV summary
CREATE OR REPLACE VIEW v_cohort_ltv AS
SELECT
    cohort_month,
    COUNT(DISTINCT customer_id)             AS customers,
    AVG(frequency)                          AS avg_frequency,
    AVG(monetary_avg)                       AS avg_order_value,
    AVG(monetary_total)                     AS avg_total_spend,
    AVG(ltv_36m)                            AS avg_predicted_ltv_36m,
    PERCENTILE_CONT(0.5) WITHIN GROUP
        (ORDER BY ltv_36m)                  AS median_predicted_ltv_36m,
    AVG(actual_ltv_12m)                     AS avg_actual_ltv_12m,
    COUNT(DISTINCT segment)                 AS segment_count
FROM v_customer_summary
WHERE cohort_month IS NOT NULL
GROUP BY cohort_month
ORDER BY cohort_month;

-- LTV segment distribution
CREATE OR REPLACE VIEW v_segment_distribution AS
SELECT
    segment,
    COUNT(*)                                    AS customer_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_customers,
    AVG(ltv_36m)                                AS avg_ltv_36m,
    SUM(ltv_36m)                                AS total_ltv_36m,
    AVG(recommended_max_cac)                    AS avg_max_cac
FROM v_customer_summary
WHERE segment IS NOT NULL
GROUP BY segment;

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Updated-at trigger
CREATE OR REPLACE FUNCTION trigger_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_customers_updated_at ON customers;
CREATE TRIGGER set_customers_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION trigger_set_updated_at();

-- Lookalike search function (used by API)
CREATE OR REPLACE FUNCTION find_lookalikes(
    query_customer_id TEXT,
    model_ver         TEXT DEFAULT 'v1',
    top_n             INTEGER DEFAULT 10
)
RETURNS TABLE (
    candidate_customer_id TEXT,
    similarity            FLOAT,
    ltv_36m               NUMERIC,
    segment               TEXT
) AS $$
DECLARE
    query_embedding vector(64);
BEGIN
    SELECT embedding INTO query_embedding
    FROM customer_embeddings
    WHERE customer_id = query_customer_id
      AND model_version = model_ver
    LIMIT 1;

    IF query_embedding IS NULL THEN
        RAISE EXCEPTION 'No embedding found for customer % / model %',
            query_customer_id, model_ver;
    END IF;

    RETURN QUERY
    SELECT
        ce.customer_id,
        1 - (ce.embedding <=> query_embedding)    AS similarity,
        p.ltv_36m,
        p.segment
    FROM customer_embeddings ce
    LEFT JOIN ltv_predictions p
        ON p.customer_id = ce.customer_id
    WHERE ce.customer_id   <> query_customer_id
      AND ce.model_version  = model_ver
    ORDER BY ce.embedding <=> query_embedding
    LIMIT top_n;
END;
$$ LANGUAGE plpgsql;