-- Migration 004 - Causal ML + Cold-Start LTV tables
BEGIN;

-- ============================================================
-- CAUSAL MODEL REGISTRY
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_model_registry (
    id                      BIGSERIAL       PRIMARY KEY,
    model_version           TEXT            NOT NULL UNIQUE,
    trained_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    observation_end         DATE,
    dataset                 TEXT            NOT NULL DEFAULT 'uci_online_retail',

    -- Model config
    outcome_variable        TEXT            NOT NULL DEFAULT 'ltv_36m',
    estimator_type          TEXT            NOT NULL DEFAULT 'DML',  -- DML | CausalForest
    n_treatments            INTEGER,
    n_controls              INTEGER,
    n_customers             INTEGER,

    -- Fit quality
    nuisance_cv_folds       INTEGER,
    final_stage_r2          NUMERIC(8,  6),
    ate_estimate            NUMERIC(14, 4),  -- average treatment effect
    ate_stderr              NUMERIC(14, 4),

    -- Tracking
    wandb_run_id            TEXT,
    pipeline_run_id         TEXT,

    CHECK (estimator_type IN ('DML', 'CausalForest')),
    CHECK (n_treatments IS NULL OR n_treatments >= 0),
    CHECK (n_controls IS NULL OR n_controls >= 0),
    CHECK (n_customers IS NULL OR n_customers >= 0)
);

-- ============================================================
-- CAUSAL TREATMENT EFFECTS  (one row per treatment variable)
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_treatment_effects (
    id                      BIGSERIAL       PRIMARY KEY,
    model_version           TEXT            NOT NULL,
    treatment_name          TEXT            NOT NULL,
    treatment_type          TEXT            NOT NULL DEFAULT 'binary',  -- binary | continuous | categorical

    -- Average Treatment Effect
    ate                     NUMERIC(14, 4),
    ate_lower_ci            NUMERIC(14, 4),
    ate_upper_ci            NUMERIC(14, 4),
    ate_stderr              NUMERIC(14, 4),
    ate_pvalue              NUMERIC(8,  6),

    -- Effect description
    effect_description      TEXT,
    effect_direction        TEXT,           -- positive | negative | neutral
    is_significant          BOOLEAN,        -- p < 0.05

    -- CATE stats
    cate_mean               NUMERIC(14, 4),
    cate_std                NUMERIC(12, 4),
    cate_min                NUMERIC(14, 4),
    cate_max                NUMERIC(14, 4),

    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (model_version, treatment_name),
    CHECK (treatment_type IN ('binary', 'continuous', 'categorical')),
    CHECK (effect_direction IS NULL OR effect_direction IN ('positive', 'negative', 'neutral')),
    CHECK (ate_pvalue IS NULL OR (ate_pvalue >= 0 AND ate_pvalue <= 1)),
    CHECK (ate_lower_ci IS NULL OR ate_upper_ci IS NULL OR ate_lower_ci <= ate_upper_ci)
);

-- ============================================================
-- PER-CUSTOMER CATE  (Conditional Average Treatment Effect)
-- ============================================================
CREATE TABLE IF NOT EXISTS customer_cate (
    id                      BIGSERIAL       PRIMARY KEY,
    customer_id             TEXT            NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_version           TEXT            NOT NULL,
    treatment_name          TEXT            NOT NULL,

    cate_estimate           NUMERIC(14, 4),
    cate_lower              NUMERIC(14, 4),
    cate_upper              NUMERIC(14, 4),

    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE (customer_id, model_version, treatment_name),
    CHECK (cate_lower IS NULL OR cate_upper IS NULL OR cate_lower <= cate_upper)
);

CREATE INDEX IF NOT EXISTS idx_cate_customer
    ON customer_cate(customer_id);
CREATE INDEX IF NOT EXISTS idx_cate_treatment
    ON customer_cate(treatment_name);
CREATE INDEX IF NOT EXISTS idx_cate_model_version
    ON customer_cate(model_version);

-- ============================================================
-- FIRMOGRAPHIC COLD-START LTV TABLE
-- ============================================================
-- Already defined in schema.sql but we extend it here
ALTER TABLE firmographic_ltv
    ADD COLUMN IF NOT EXISTS n_customers          INTEGER,
    ADD COLUMN IF NOT EXISTS causal_model_version TEXT,
    ADD COLUMN IF NOT EXISTS feature_importances  JSONB,
    ADD COLUMN IF NOT EXISTS percentile_rank      NUMERIC(6, 2);

-- ============================================================
-- CAUSAL DAG METADATA
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_dag_nodes (
    id              BIGSERIAL   PRIMARY KEY,
    model_version   TEXT        NOT NULL,
    node_name       TEXT        NOT NULL,
    node_type       TEXT        NOT NULL,  -- treatment | outcome | confounder | instrument
    description     TEXT,
    UNIQUE (model_version, node_name),
    CHECK (node_type IN ('treatment', 'outcome', 'confounder', 'instrument'))
);

CREATE TABLE IF NOT EXISTS causal_dag_edges (
    id              BIGSERIAL   PRIMARY KEY,
    model_version   TEXT        NOT NULL,
    from_node       TEXT        NOT NULL,
    to_node         TEXT        NOT NULL,
    edge_type       TEXT        DEFAULT 'causal',
    UNIQUE (model_version, from_node, to_node),
    CHECK (edge_type IN ('causal', 'associative', 'derived')),
    CHECK (from_node <> to_node)
);

-- ============================================================
-- CAUSAL LEVER RECOMMENDATIONS  (per customer)
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_lever_recommendations (
    id                      BIGSERIAL       PRIMARY KEY,
    customer_id             TEXT            NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
    model_version           TEXT            NOT NULL,
    top_lever               TEXT,
    top_lever_effect_usd    NUMERIC(14, 4),
    lever_json              JSONB,          -- [{lever, effect, description}, ...]
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (customer_id, model_version),
    CHECK (top_lever_effect_usd IS NULL OR top_lever_effect_usd >= 0)
);

CREATE INDEX IF NOT EXISTS idx_causal_treatment_effects_model_version
    ON causal_treatment_effects(model_version);
CREATE INDEX IF NOT EXISTS idx_causal_lever_reco_customer
    ON causal_lever_recommendations(customer_id);
CREATE INDEX IF NOT EXISTS idx_causal_lever_reco_model_version
    ON causal_lever_recommendations(model_version);
CREATE INDEX IF NOT EXISTS idx_causal_dag_nodes_model_version
    ON causal_dag_nodes(model_version);
CREATE INDEX IF NOT EXISTS idx_causal_dag_edges_model_version
    ON causal_dag_edges(model_version);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_causal_treatment_effects_model_version'
    ) THEN
        ALTER TABLE causal_treatment_effects
            ADD CONSTRAINT fk_causal_treatment_effects_model_version
            FOREIGN KEY (model_version)
            REFERENCES causal_model_registry(model_version)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_customer_cate_model_version'
    ) THEN
        ALTER TABLE customer_cate
            ADD CONSTRAINT fk_customer_cate_model_version
            FOREIGN KEY (model_version)
            REFERENCES causal_model_registry(model_version)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_customer_cate_treatment'
    ) THEN
        ALTER TABLE customer_cate
            ADD CONSTRAINT fk_customer_cate_treatment
            FOREIGN KEY (model_version, treatment_name)
            REFERENCES causal_treatment_effects(model_version, treatment_name)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_causal_dag_nodes_model_version'
    ) THEN
        ALTER TABLE causal_dag_nodes
            ADD CONSTRAINT fk_causal_dag_nodes_model_version
            FOREIGN KEY (model_version)
            REFERENCES causal_model_registry(model_version)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_causal_dag_edges_model_version'
    ) THEN
        ALTER TABLE causal_dag_edges
            ADD CONSTRAINT fk_causal_dag_edges_model_version
            FOREIGN KEY (model_version)
            REFERENCES causal_model_registry(model_version)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_causal_dag_edges_from_node'
    ) THEN
        ALTER TABLE causal_dag_edges
            ADD CONSTRAINT fk_causal_dag_edges_from_node
            FOREIGN KEY (model_version, from_node)
            REFERENCES causal_dag_nodes(model_version, node_name)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_causal_dag_edges_to_node'
    ) THEN
        ALTER TABLE causal_dag_edges
            ADD CONSTRAINT fk_causal_dag_edges_to_node
            FOREIGN KEY (model_version, to_node)
            REFERENCES causal_dag_nodes(model_version, node_name)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_causal_lever_recommendations_model_version'
    ) THEN
        ALTER TABLE causal_lever_recommendations
            ADD CONSTRAINT fk_causal_lever_recommendations_model_version
            FOREIGN KEY (model_version)
            REFERENCES causal_model_registry(model_version)
            ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_firmographic_ltv_causal_model_version'
    ) THEN
        ALTER TABLE firmographic_ltv
            ADD CONSTRAINT fk_firmographic_ltv_causal_model_version
            FOREIGN KEY (causal_model_version)
            REFERENCES causal_model_registry(model_version)
            ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'firmographic_ltv_n_customers_nonnegative'
    ) THEN
        ALTER TABLE firmographic_ltv
            ADD CONSTRAINT firmographic_ltv_n_customers_nonnegative
            CHECK (n_customers IS NULL OR n_customers >= 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'firmographic_ltv_percentile_rank_range'
    ) THEN
        ALTER TABLE firmographic_ltv
            ADD CONSTRAINT firmographic_ltv_percentile_rank_range
            CHECK (percentile_rank IS NULL OR (percentile_rank >= 0 AND percentile_rank <= 100));
    END IF;
END $$;

-- ============================================================
-- VIEWS
-- ============================================================

-- Treatment effects sorted by absolute ATE
CREATE OR REPLACE VIEW v_top_causal_levers AS
SELECT
    model_version,
    treatment_name,
    ate,
    ate_lower_ci,
    ate_upper_ci,
    effect_description,
    effect_direction,
    is_significant,
    cate_std,
    ABS(ate) AS abs_ate
FROM causal_treatment_effects;

-- Cold-start lookup with segment assignment
CREATE OR REPLACE VIEW v_coldstart_segments AS
SELECT
    vertical,
    company_size,
    channel,
    plan_tier,
    ltv_36m_estimate,
    ci_lower,
    ci_upper,
    CASE
        WHEN ltv_36m_estimate > 10000 THEN 'champions'
        WHEN ltv_36m_estimate > 5000  THEN 'high_value'
        WHEN ltv_36m_estimate > 1000  THEN 'medium_value'
        ELSE 'low_value'
    END AS predicted_segment,
    ROUND(0.40 * ltv_36m_estimate, 2) AS recommended_max_cac,
    computed_at
FROM firmographic_ltv;

COMMIT;
