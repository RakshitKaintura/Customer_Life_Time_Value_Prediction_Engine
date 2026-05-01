-- Migration 006 — Fusion data-quality backfill and view hardening
BEGIN;

-- Backfill component predictions where they were not persisted.
UPDATE final_ltv_scores f
SET bgnbd_ltv_12m = COALESCE(f.bgnbd_ltv_12m, b.ltv_12m),
    bgnbd_ltv_36m = COALESCE(f.bgnbd_ltv_36m, b.ltv_36m),
    transformer_ltv_12m = COALESCE(f.transformer_ltv_12m, t.ltv_12m),
    transformer_ltv_36m = COALESCE(f.transformer_ltv_36m, t.ltv_36m)
FROM bgnbd_predictions b
JOIN transformer_predictions t ON t.customer_id = b.customer_id
WHERE f.customer_id = b.customer_id
  AND f.ltv_source = 'full_model';

-- Fill any remaining BG/NBD nulls with safe defaults for comparison views.
UPDATE final_ltv_scores
SET bgnbd_ltv_12m = COALESCE(bgnbd_ltv_12m, 0::numeric),
    bgnbd_ltv_36m = COALESCE(bgnbd_ltv_36m, bgnbd_ltv_12m * 2.5, 0::numeric)
WHERE ltv_source = 'full_model';

-- Enforce monotonic horizons and aligned 24m interpolation.
UPDATE final_ltv_scores
SET ltv_36m = GREATEST(ltv_36m, ltv_12m),
    ltv_24m = sqrt(GREATEST(ltv_36m, ltv_12m) * ltv_12m)
WHERE ltv_source = 'full_model'
  AND ltv_12m IS NOT NULL
  AND ltv_36m IS NOT NULL;

-- Recompute segment and CAC from corrected 36m values.
UPDATE final_ltv_scores
SET segment = CASE
        WHEN ltv_36m > 10000 THEN 'champions'
        WHEN ltv_36m > 5000 THEN 'high_value'
        WHEN ltv_36m > 1000 THEN 'medium_value'
        ELSE 'low_value'
    END,
    recommended_max_cac = CASE
        WHEN ltv_36m > 10000 THEN ltv_36m * 0.50
        WHEN ltv_36m > 5000 THEN ltv_36m * 0.40
        WHEN ltv_36m > 1000 THEN ltv_36m * 0.30
        ELSE ltv_36m * 0.20
    END
WHERE ltv_source = 'full_model';

-- Recompute empirical percentile in [0,100].
WITH ranked AS (
    SELECT id,
           ROUND((PERCENT_RANK() OVER (ORDER BY ltv_36m)) * 100)::int AS pct
    FROM final_ltv_scores
    WHERE ltv_source = 'full_model'
)
UPDATE final_ltv_scores f
SET ltv_percentile = r.pct
FROM ranked r
WHERE f.id = r.id;

-- Backfill fusion model improvement metrics using observed MAE on 12m labels.
WITH latest AS (
    SELECT id
    FROM fusion_model_registry
    ORDER BY trained_at DESC
    LIMIT 1
), metrics AS (
    SELECT
        AVG(ABS(r.actual_ltv_12m - f.ltv_12m)) AS mae_fusion,
        AVG(ABS(r.actual_ltv_12m - COALESCE(f.bgnbd_ltv_12m, b.ltv_12m))) AS mae_bgnbd,
        AVG(ABS(r.actual_ltv_12m - COALESCE(f.transformer_ltv_12m, t.ltv_12m))) AS mae_trans
    FROM final_ltv_scores f
    JOIN rfm_features r ON r.customer_id = f.customer_id
    LEFT JOIN bgnbd_predictions b ON b.customer_id = f.customer_id
    LEFT JOIN transformer_predictions t ON t.customer_id = f.customer_id
    WHERE f.ltv_source = 'full_model'
      AND r.actual_ltv_12m IS NOT NULL
)
UPDATE fusion_model_registry m
SET improvement_over_bgnbd_pct =
        ((metrics.mae_bgnbd - metrics.mae_fusion) / NULLIF(metrics.mae_bgnbd, 0)) * 100,
    improvement_over_transformer_pct =
        ((metrics.mae_trans - metrics.mae_fusion) / NULLIF(metrics.mae_trans, 0)) * 100
FROM latest, metrics
WHERE m.id = latest.id;

-- Harden comparison view against null components.
CREATE OR REPLACE VIEW v_model_fusion_comparison AS
SELECT
    customer_id,
    COALESCE(bgnbd_ltv_36m, 0::numeric(14,4)) AS bgnbd_ltv_36m,
    COALESCE(transformer_ltv_36m, 0::numeric(14,4)) AS transformer_ltv_36m,
    ltv_36m AS fusion_ltv_36m,
    ltv_36m - COALESCE(bgnbd_ltv_36m, 0::numeric(14,4)) AS fusion_vs_bgnbd,
    ltv_36m - COALESCE(transformer_ltv_36m, 0::numeric(14,4)) AS fusion_vs_transformer,
    segment,
    ltv_percentile
FROM final_ltv_scores
WHERE ltv_source = 'full_model';

COMMIT;
