"""
DuckDB aggregation layer.

Runs fast analytical SQL directly on Parquet / CSV snapshots
before handing results to the Polars feature pipeline.
DuckDB operates fully in-process — no server required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from loguru import logger

from backend.config import settings


class DuckDBAggregator:
    """
    Manages a persistent DuckDB connection and exposes
    pre-built aggregation queries for the LTV pipeline.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        postgres_url: str | None = None,
    ) -> None:
        """
        Args:
            db_path:       DuckDB database path. Defaults to in-memory.
            postgres_url:  Optional Postgres connection string for reading
                           directly from Supabase via postgres_scanner.
        """
        self._db_path = str(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._postgres_url = postgres_url or settings.DATABASE_URL

    # ─────────────────────────────────────────────────────────
    # Connection management
    # ─────────────────────────────────────────────────────────

    def connect(self) -> "DuckDBAggregator":
        self._conn = duckdb.connect(self._db_path)
        # Install extensions once
        self._conn.execute("INSTALL httpfs; LOAD httpfs;")
        self._conn.execute("INSTALL postgres; LOAD postgres;")
        logger.info("DuckDB connected — {}", self._db_path)
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "DuckDBAggregator":
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("Call .connect() or use as a context manager first.")
        return self._conn

    # ─────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────

    def register_parquet(self, name: str, path: str | Path) -> None:
        """Register a Parquet file as a DuckDB view."""
        self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{path}')")
        logger.debug("Registered Parquet view '{}' ← {}", name, path)

    def register_csv(
        self,
        name: str,
        path: str | Path,
        has_header: bool = True,
        delimiter: str = ",",
    ) -> None:
        """Register a CSV file as a DuckDB view."""
        self.conn.execute(
            f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * FROM read_csv_auto('{path}', header={str(has_header).lower()},
                                        delim='{delimiter}', sample_size=-1)
            """
        )
        logger.debug("Registered CSV view '{}' ← {}", name, path)

    def register_polars(self, name: str, df: pl.DataFrame) -> None:
        """Register a Polars DataFrame as a DuckDB relation."""
        # DuckDB can read Arrow directly from Polars
        arrow = df.to_arrow()
        self.conn.register(name, arrow)
        logger.debug("Registered Polars DF as DuckDB view '{}' ({} rows)", name, len(df))

    def attach_postgres(self, alias: str = "pg") -> None:
        """
        Attach Supabase Postgres so DuckDB can query it directly.
        Requires the postgres extension.
        """
        if not self._postgres_url:
            raise ValueError("postgres_url is not configured.")
        self.conn.execute(
            f"ATTACH '{self._postgres_url}' AS {alias} (TYPE POSTGRES, READ_ONLY)"
        )
        logger.info("Attached Postgres as DuckDB alias '{}'", alias)

    # ─────────────────────────────────────────────────────────
    # Query → Polars
    # ─────────────────────────────────────────────────────────

    def query(self, sql: str) -> pl.DataFrame:
        """Execute SQL and return a Polars DataFrame."""
        result = self.conn.execute(sql)
        return pl.from_arrow(result.arrow())

    def query_params(self, sql: str, params: list[Any]) -> pl.DataFrame:
        """Execute parametrised SQL and return a Polars DataFrame."""
        result = self.conn.execute(sql, params)
        return pl.from_arrow(result.arrow())

    # ─────────────────────────────────────────────────────────
    # Pre-built aggregations
    # ─────────────────────────────────────────────────────────

    def agg_basic_stats(self, view: str = "transactions") -> pl.DataFrame:
        """Row counts, date range, customer count, revenue totals."""
        return self.query(
            f"""
            SELECT
                COUNT(*)                                        AS total_rows,
                COUNT(DISTINCT customer_id)                     AS unique_customers,
                COUNT(DISTINCT invoice_no)                      AS unique_invoices,
                COUNT(DISTINCT stock_code)                      AS unique_products,
                COUNT(DISTINCT country)                         AS unique_countries,
                MIN(invoice_date)                               AS earliest_date,
                MAX(invoice_date)                               AS latest_date,
                SUM(quantity * unit_price)                      AS total_revenue,
                AVG(quantity * unit_price)                      AS avg_order_value,
                PERCENTILE_CONT(0.5) WITHIN GROUP
                    (ORDER BY quantity * unit_price)            AS median_order_value,
                STDDEV(quantity * unit_price)                   AS std_order_value
            FROM {view}
            WHERE quantity > 0
              AND unit_price > 0
              AND customer_id IS NOT NULL
            """
        )

    def agg_monthly_revenue(self, view: str = "transactions") -> pl.DataFrame:
        """Monthly revenue and order counts."""
        return self.query(
            f"""
            SELECT
                DATE_TRUNC('month', invoice_date)               AS month,
                COUNT(DISTINCT customer_id)                     AS active_customers,
                COUNT(DISTINCT invoice_no)                      AS orders,
                SUM(quantity * unit_price)                      AS revenue,
                AVG(quantity * unit_price)                      AS avg_order_value,
                COUNT(DISTINCT stock_code)                      AS unique_products_sold
            FROM {view}
            WHERE quantity > 0
              AND unit_price > 0
              AND customer_id IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
        )

    def agg_customer_totals(self, view: str = "transactions") -> pl.DataFrame:
        """Per-customer totals for seeding the customers table."""
        return self.query(
            f"""
            SELECT
                customer_id,
                COUNT(DISTINCT invoice_no)                      AS total_orders,
                SUM(quantity * unit_price)                      AS total_revenue,
                MIN(invoice_date)                               AS first_purchase_date,
                MAX(invoice_date)                               AS last_purchase_date,
                COUNT(DISTINCT country)                         AS countries_count,
                MODE(country)                                   AS primary_country,
                AVG(quantity * unit_price)                      AS avg_order_value,
                COUNT(DISTINCT stock_code)                      AS unique_products
            FROM {view}
            WHERE quantity > 0
              AND unit_price > 0
              AND customer_id IS NOT NULL
            GROUP BY customer_id
            """
        )

    def agg_product_categories(self, view: str = "transactions") -> pl.DataFrame:
        """Revenue and order count by product category."""
        return self.query(
            f"""
            SELECT
                COALESCE(product_category, 'unknown')           AS category,
                COUNT(DISTINCT customer_id)                     AS customers,
                COUNT(DISTINCT invoice_no)                      AS orders,
                SUM(quantity * unit_price)                      AS revenue,
                AVG(quantity * unit_price)                      AS avg_order_value
            FROM {view}
            WHERE quantity > 0
              AND unit_price > 0
              AND customer_id IS NOT NULL
            GROUP BY 1
            ORDER BY revenue DESC
            """
        )

    def agg_country_revenue(self, view: str = "transactions") -> pl.DataFrame:
        """Revenue by country."""
        return self.query(
            f"""
            SELECT
                country,
                COUNT(DISTINCT customer_id)                     AS customers,
                SUM(quantity * unit_price)                      AS revenue,
                AVG(quantity * unit_price)                      AS avg_order_value
            FROM {view}
            WHERE quantity > 0
              AND unit_price > 0
              AND customer_id IS NOT NULL
              AND country IS NOT NULL
            GROUP BY country
            ORDER BY revenue DESC
            """
        )

    def agg_rfm_base(
        self,
        view: str = "transactions",
        observation_end: str = "2011-06-30",
    ) -> pl.DataFrame:
        """
        Compute the BG/NBD model inputs directly in DuckDB.

        Returns one row per customer with:
          - frequency (x)   : number of repeat purchase *periods* (invoices - 1)
          - recency  (t_x)  : days from first to last purchase
          - T        (T)    : days from first purchase to observation_end
          - monetary_avg     : mean of per-invoice totals (for Gamma-Gamma)
        """
        return self.query(
            f"""
            WITH invoice_totals AS (
                SELECT
                    customer_id,
                    invoice_no,
                    invoice_date::DATE                          AS purchase_date,
                    SUM(quantity * unit_price)                  AS invoice_total
                FROM {view}
                WHERE quantity > 0
                  AND unit_price > 0
                  AND customer_id IS NOT NULL
                  AND invoice_date::DATE <= '{observation_end}'
                GROUP BY customer_id, invoice_no, invoice_date::DATE
            ),
            customer_invoices AS (
                SELECT
                    customer_id,
                    purchase_date,
                    invoice_total,
                    ROW_NUMBER() OVER (
                        PARTITION BY customer_id ORDER BY purchase_date
                    )                                           AS purchase_rank,
                    COUNT(*) OVER (PARTITION BY customer_id)   AS total_invoices
                FROM invoice_totals
            )
            SELECT
                customer_id,
                -- BG/NBD inputs (standard notation)
                (COUNT(DISTINCT purchase_date) - 1)             AS frequency,
                DATEDIFF('day', MIN(purchase_date),
                                MAX(purchase_date))             AS recency_days,
                DATEDIFF('day', MIN(purchase_date),
                                '{observation_end}')            AS t_days,
                -- Gamma-Gamma inputs
                AVG(invoice_total)                              AS monetary_avg,
                STDDEV(invoice_total)                           AS monetary_std,
                SUM(invoice_total)                              AS monetary_total,
                VARIANCE(invoice_total)                         AS purchase_variance,
                -- Extra counts
                COUNT(DISTINCT purchase_date)                   AS orders_count,
                MIN(purchase_date)                              AS first_purchase_date,
                MAX(purchase_date)                              AS last_purchase_date
            FROM customer_invoices
            GROUP BY customer_id
            HAVING COUNT(DISTINCT purchase_date) >= 1
            """
        )

    def agg_inter_purchase_times(self, view: str = "transactions") -> pl.DataFrame:
        """Compute per-customer average and std of days between purchases."""
        return self.query(
            f"""
            WITH daily_orders AS (
                SELECT
                    customer_id,
                    invoice_date::DATE                              AS purchase_date
                FROM {view}
                WHERE quantity > 0 AND unit_price > 0
                  AND customer_id IS NOT NULL
                GROUP BY customer_id, invoice_date::DATE
            ),
            lagged AS (
                SELECT
                    customer_id,
                    purchase_date,
                    LAG(purchase_date) OVER (
                        PARTITION BY customer_id ORDER BY purchase_date
                    )                                               AS prev_date
                FROM daily_orders
            )
            SELECT
                customer_id,
                AVG(DATEDIFF('day', prev_date, purchase_date))      AS avg_days_between_orders,
                STDDEV(DATEDIFF('day', prev_date, purchase_date))   AS std_days_between_orders,
                MIN(DATEDIFF('day', prev_date, purchase_date))      AS min_days_between_orders,
                MAX(DATEDIFF('day', prev_date, purchase_date))      AS max_days_between_orders
            FROM lagged
            WHERE prev_date IS NOT NULL
            GROUP BY customer_id
            """
        )

    def agg_first_second_purchase(self, view: str = "transactions") -> pl.DataFrame:
        """Days between first and second purchase — strong early churn signal."""
        return self.query(
            f"""
            WITH ranked AS (
                SELECT
                    customer_id,
                    invoice_date::DATE                          AS purchase_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY customer_id ORDER BY invoice_date
                    )                                           AS rn
                FROM {view}
                WHERE quantity > 0 AND unit_price > 0
                  AND customer_id IS NOT NULL
                GROUP BY customer_id, invoice_date::DATE
            )
            SELECT
                r1.customer_id,
                r1.purchase_date                                AS first_purchase_date,
                r2.purchase_date                                AS second_purchase_date,
                DATEDIFF('day', r1.purchase_date,
                                r2.purchase_date)               AS days_to_second_purchase
            FROM ranked r1
            LEFT JOIN ranked r2
                ON r1.customer_id = r2.customer_id
               AND r2.rn = 2
            WHERE r1.rn = 1
            """
        )

    def agg_cohort_sizes(self, view: str = "transactions") -> pl.DataFrame:
        """Monthly acquisition cohort sizes."""
        return self.query(
            f"""
            WITH first_purchases AS (
                SELECT
                    customer_id,
                    DATE_TRUNC('month', MIN(invoice_date))::DATE    AS cohort_month
                FROM {view}
                WHERE quantity > 0 AND unit_price > 0
                  AND customer_id IS NOT NULL
                GROUP BY customer_id
            )
            SELECT
                TO_CHAR(cohort_month, 'YYYY-MM')                    AS cohort_month,
                COUNT(*)                                            AS cohort_size
            FROM first_purchases
            GROUP BY cohort_month
            ORDER BY cohort_month
            """
        )

    def agg_cohort_retention(
        self,
        view: str = "transactions",
        max_months: int = 12,
    ) -> pl.DataFrame:
        """
        Cohort retention matrix — % of each monthly cohort still purchasing
        in months 1, 2, … max_months after acquisition.
        """
        return self.query(
            f"""
            WITH cohort_base AS (
                SELECT
                    customer_id,
                    DATE_TRUNC('month', MIN(invoice_date))::DATE    AS cohort_date
                FROM {view}
                WHERE quantity > 0 AND unit_price > 0
                  AND customer_id IS NOT NULL
                GROUP BY customer_id
            ),
            monthly_activity AS (
                SELECT DISTINCT
                    t.customer_id,
                    DATE_TRUNC('month', t.invoice_date)::DATE       AS activity_month
                FROM {view} t
                WHERE quantity > 0 AND unit_price > 0
                  AND customer_id IS NOT NULL
            ),
            joined AS (
                SELECT
                    cb.customer_id,
                    TO_CHAR(cb.cohort_date, 'YYYY-MM')              AS cohort_month,
                    DATEDIFF('month', cb.cohort_date,
                              ma.activity_month)                     AS months_since_first
                FROM cohort_base cb
                JOIN monthly_activity ma USING (customer_id)
                WHERE DATEDIFF('month', cb.cohort_date,
                               ma.activity_month) BETWEEN 0 AND {max_months}
            )
            SELECT
                cohort_month,
                months_since_first,
                COUNT(DISTINCT customer_id)                          AS active_customers
            FROM joined
            GROUP BY cohort_month, months_since_first
            ORDER BY cohort_month, months_since_first
            """
        )

    def compute_amount_buckets(
        self,
        view: str = "transactions",
        n_buckets: int = 5,
    ) -> pl.DataFrame:
        """Compute quantile boundaries for amount bucketing."""
        quantiles = ", ".join(
            f"QUANTILE_CONT(quantity * unit_price, {i / n_buckets}) AS q{i}"
            for i in range(1, n_buckets + 1)
        )
        return self.query(
            f"""
            SELECT {quantiles}
            FROM {view}
            WHERE quantity > 0 AND unit_price > 0
              AND customer_id IS NOT NULL
            """
        )