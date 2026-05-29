-- Cohort retention matrix for the dashboard.
-- max_months is optional; when omitted, the function uses the observed data span.

CREATE OR REPLACE FUNCTION get_cohort_retention_matrix(max_months INTEGER DEFAULT NULL)
RETURNS TABLE (
    cohort_month        TEXT,
    months_since_first  INTEGER,
    active_customers    BIGINT,
    cohort_n            BIGINT,
    retention_rate_pct  NUMERIC
)
LANGUAGE SQL
STABLE
AS $$
WITH cohort_base AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(invoice_date))::DATE AS cohort_date
    FROM transactions
    WHERE quantity > 0
      AND unit_price > 0
      AND customer_id IS NOT NULL
    GROUP BY customer_id
),
cohort_sizes AS (
    SELECT
        cohort_date,
        COUNT(*) AS cohort_n
    FROM cohort_base
    GROUP BY cohort_date
),
monthly_activity AS (
    SELECT DISTINCT
        customer_id,
        DATE_TRUNC('month', invoice_date)::DATE AS activity_month
    FROM transactions
    WHERE quantity > 0
      AND unit_price > 0
      AND customer_id IS NOT NULL
),
observed_span AS (
    SELECT
        MAX(
            (
                EXTRACT(YEAR FROM AGE(activity_month, cohort_date)) * 12
              + EXTRACT(MONTH FROM AGE(activity_month, cohort_date))
            )::INTEGER
        ) AS max_months_since_first
    FROM cohort_base
    JOIN monthly_activity USING (customer_id)
),
joined AS (
    SELECT
        cohort_base.cohort_date,
        (
            EXTRACT(YEAR FROM AGE(monthly_activity.activity_month, cohort_base.cohort_date)) * 12
          + EXTRACT(MONTH FROM AGE(monthly_activity.activity_month, cohort_base.cohort_date))
        )::INTEGER AS months_since_first,
        monthly_activity.customer_id
    FROM cohort_base
    JOIN monthly_activity USING (customer_id)
    CROSS JOIN observed_span
    WHERE (
            EXTRACT(YEAR FROM AGE(monthly_activity.activity_month, cohort_base.cohort_date)) * 12
          + EXTRACT(MONTH FROM AGE(monthly_activity.activity_month, cohort_base.cohort_date))
        )::INTEGER BETWEEN 0 AND COALESCE(max_months, observed_span.max_months_since_first)
)
SELECT
    TO_CHAR(joined.cohort_date, 'YYYY-MM') AS cohort_month,
    joined.months_since_first,
    COUNT(DISTINCT joined.customer_id) AS active_customers,
    cohort_sizes.cohort_n,
    ROUND(
        100.0 * COUNT(DISTINCT joined.customer_id)::NUMERIC
        / NULLIF(cohort_sizes.cohort_n, 0),
        2
    ) AS retention_rate_pct
FROM joined
JOIN cohort_sizes USING (cohort_date)
GROUP BY joined.cohort_date, joined.months_since_first, cohort_sizes.cohort_n
ORDER BY joined.cohort_date, joined.months_since_first;
$$;
