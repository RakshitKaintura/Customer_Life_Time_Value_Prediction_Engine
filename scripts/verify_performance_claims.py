"""
Verify core performance claims with reproducible measurements.

Checks:
1) Real-world scoring latency from DB records (p50/p95/max).
2) Pandas vs Polars in-memory footprint on the same raw CSV.

Usage:
  .venv\\Scripts\\python.exe scripts/verify_performance_claims.py
  .venv\\Scripts\\python.exe scripts/verify_performance_claims.py --csv backend/data/raw/OnlineRetail.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
from sqlalchemy import text

from backend.db.supabase_client import get_db_engine


def latency_report() -> dict[str, float | int | None]:
    engine = get_db_engine()
    query = text(
        """
        SELECT
          COUNT(*)                                                        AS n_rows,
          PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY scoring_latency_ms) AS p50_ms,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY scoring_latency_ms) AS p95_ms,
          MAX(scoring_latency_ms)                                         AS max_ms
        FROM final_ltv_scores
        WHERE ltv_source = 'full_model'
          AND scoring_latency_ms IS NOT NULL
        """
    )
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    return dict(row) if row else {"n_rows": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}


def memory_report(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    pdf = pd.read_csv(csv_path, encoding="ISO-8859-1")
    p_mem_mb = float(pdf.memory_usage(deep=True).sum()) / (1024 * 1024)

    plf = pl.read_csv(
        csv_path,
        encoding="latin1",
        infer_schema_length=10000,
        schema_overrides={"InvoiceNo": pl.Utf8},
    )
    pl_mem_mb = float(plf.estimated_size("mb"))

    reduction_pct = ((p_mem_mb - pl_mem_mb) / p_mem_mb * 100.0) if p_mem_mb > 0 else 0.0
    return {
        "pandas_mb": p_mem_mb,
        "polars_mb": pl_mem_mb,
        "reduction_pct": reduction_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("backend/data/raw/OnlineRetail.csv"),
        help="Path to CSV for Pandas vs Polars memory comparison.",
    )
    args = parser.parse_args()

    print("=== Latency (DB) ===")
    lat = latency_report()
    print(lat)

    print("\n=== Memory (CSV load) ===")
    mem = memory_report(args.csv)
    print(mem)


if __name__ == "__main__":
    main()
