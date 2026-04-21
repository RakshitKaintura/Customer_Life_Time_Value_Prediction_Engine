"""
UCI Online Retail Dataset — Loader & Ingestion Pipeline.

Steps:
  1. Download from Kaggle (or load from local path)
  2. Parse and validate with Polars
  3. Clean (remove returns, nulls, zero-price)
  4. Upsert raw rows → Supabase raw_transactions
  5. Upsert cleaned rows → Supabase transactions
  6. Seed the customers table from transaction aggregates
  7. Log run to pipeline_runs

Run:
    python -m backend.data.load_data --csv-path ./data/raw/online_retail.csv
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import duckdb
import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from backend.config import settings
from backend.db.supabase_client import SupabaseClient
from backend.features.rfm import (
    assign_amount_buckets,
    assign_product_categories,
    clean_transactions,
)

app = typer.Typer(help="Load UCI Online Retail data into Supabase.")
console = Console()


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

COLUMN_RENAME_MAP = {
    "InvoiceNo":   "invoice_no",
    "StockCode":   "stock_code",
    "Description": "description",
    "Quantity":    "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice":   "unit_price",
    "CustomerID":  "customer_id",
    "Country":     "country",
}

SCHEMA_OVERRIDES = {
    "InvoiceNo":   pl.Utf8,
    "StockCode":   pl.Utf8,
    "Description": pl.Utf8,
    "Quantity":    pl.Int64,
    "UnitPrice":   pl.Float64,
    "CustomerID":  pl.Utf8,
    "Country":     pl.Utf8,
}


def load_uci_csv(path: str | Path) -> pl.DataFrame:
    """
    Load the UCI Online Retail dataset from a CSV or XLSX path.
    Handles both the original .xlsx and commonly converted .csv variants.
    """
    path = Path(path)
    logger.info("Loading UCI dataset from: {}", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download from https://www.kaggle.com/datasets/vijayuv/onlineretail"
        )

    if path.suffix.lower() == ".xlsx":
        # Polars can read Excel via calamine
        df = pl.read_excel(
            path,
            schema_overrides={k: v for k, v in SCHEMA_OVERRIDES.items()},
        )
    else:
        # ── THE NUCLEAR OPTION: DUCKDB LOADER ──────────────────
        # DuckDB is world-class at parsing messy CSV dates (M/D/Y H:M vs D/M/Y)
        # It handles missing leading zeros (1/1/2010) where Polars is too strict.
        logger.info("Reading CSV with DuckDB for robust date parsing...")
        rel_path = str(path).replace("\\", "/") # DuckDB prefers forward slashes
        df = duckdb.sql(f"""
            SELECT * FROM read_csv_auto('{rel_path}', 
                types={{'InvoiceNo': 'VARCHAR', 'StockCode': 'VARCHAR', 'CustomerID': 'VARCHAR'}},
                all_varchar=False,
                encoding='ISO_8859_1',
                ignore_errors=True,
                dateformat='%m/%d/%Y',
                timestampformat='%m/%d/%Y %H:%M'
            )
        """).pl()

    # Rename to snake_case
    existing_renames = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(existing_renames)

    # Normalise customer_id to string and clean up
    df = df.with_columns(
        pl.col("customer_id")
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.replace(r"\.0$", "") 
    )

    # DuckDB with explicit format should have already handled conversion.
    # We just ensure it's temporal and UTC.
    if df["invoice_date"].dtype == pl.Utf8:
        # Fallback if DuckDB didn't detect it, though with explicit formats it should.
        df = df.with_columns(pl.col("invoice_date").str.to_datetime(strict=False))

    if df["invoice_date"].dtype.is_temporal():
        df = df.with_columns(
            pl.col("invoice_date").dt.replace_time_zone("UTC")
        )


    logger.info(
        "Loaded {} rows — {} columns — date range {} to {}",
        len(df),
        len(df.columns),
        df["invoice_date"].min(),
        df["invoice_date"].max(),
    )
    return df


# ─────────────────────────────────────────────────────────────
# Seed helpers
# ─────────────────────────────────────────────────────────────

def build_customers_from_transactions(cleaned: pl.DataFrame) -> pl.DataFrame:
    """
    Derive the customers table from cleaned transaction data.
    UCI dataset has no separate customer table — we derive it.
    """
    return (
        cleaned.lazy()
        .filter(pl.col("customer_id").is_not_null())
        .group_by("customer_id")
        .agg(
            pl.col("invoice_date").min().alias("first_purchase_date"),
            pl.col("invoice_date").max().alias("last_purchase_date"),
            pl.col("invoice_no").n_unique().alias("total_orders"),
            (pl.col("quantity") * pl.col("unit_price")).sum().alias("total_revenue"),
            pl.col("country").mode().first().alias("country"),
        )
        .with_columns(
            pl.lit("organic").alias("acquisition_channel"),
            pl.lit(False).alias("is_synthetic"),
            pl.lit("uci_online_retail").alias("source_dataset"),
        )
        .collect()
    )


def build_raw_records(df: pl.DataFrame) -> list[dict]:
    """Convert raw DataFrame to list of dicts for upsert."""
    return (
        df.filter(pl.col("invoice_date").is_not_null())   # drop rows with unparseable dates
        .with_columns(
            pl.col("invoice_date").dt.to_string("%Y-%m-%dT%H:%M:%S+00:00"),
            pl.col("customer_id").cast(pl.Utf8),
        )
        .to_dicts()
    )


def build_cleaned_records(df: pl.DataFrame) -> list[dict]:
    """Convert cleaned DataFrame to list of dicts for upsert."""
    # Drop generated columns — Postgres computes these automatically
    generated_cols = [c for c in ["line_total"] if c in df.columns]
    return (
        df.filter(pl.col("invoice_date").is_not_null())   # drop rows with unparseable dates
        .drop(generated_cols)
        .with_columns(
            pl.col("invoice_date").dt.to_string("%Y-%m-%dT%H:%M:%S+00:00"),
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("amount_bucket").cast(pl.Int32),
        )
        .to_dicts()
    )




# ─────────────────────────────────────────────────────────────
# Main ingestion pipeline
# ─────────────────────────────────────────────────────────────

def run_ingestion(
    csv_path: Path,
    dry_run: bool = False,
    truncate: bool = False,
    skip_raw: bool = False,
    skip_cleaned: bool = False,
    skip_customers: bool = False,
    batch_size: int = 500,
) -> dict:
    """
    Full ingestion pipeline. Returns a summary dict.
    """
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc)
    db = SupabaseClient(use_service_role=True)

    summary: dict = {
        "run_id": run_id,
        "csv_path": str(csv_path),
        "dry_run": dry_run,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:

        # ── Step 1: Load raw data ───────────────────────────
        task = progress.add_task("Loading CSV…", total=None)
        raw_df = load_uci_csv(csv_path)
        summary["raw_rows"] = len(raw_df)
        progress.update(task, completed=1, total=1)

        # ── Step 2: Clean ───────────────────────────────────
        task2 = progress.add_task("Cleaning transactions…", total=None)
        cleaned = clean_transactions(raw_df)
        cleaned = assign_product_categories(cleaned)
        cleaned = assign_amount_buckets(cleaned)
        summary["cleaned_rows"] = len(cleaned)
        summary["removed_rows"] = len(raw_df) - len(cleaned)
        progress.update(task2, completed=1, total=1)

        if dry_run:
            console.print("[yellow]Dry run — skipping DB writes[/yellow]")
            logger.info("Dry run complete. raw={}, cleaned={}", len(raw_df), len(cleaned))
            return summary

        # ── Step 2.5: Truncate (Optional) ───────────────────
        if truncate:
            task_tr = progress.add_task("Truncating tables…", total=None)
            db.execute_sql("TRUNCATE TABLE raw_transactions, transactions, customers CASCADE;")
            logger.info("Tables truncated: raw_transactions, transactions, customers")
            progress.update(task_tr, completed=1, total=1)

        # ── Step 3: Upsert raw transactions ─────────────────
        if not skip_raw:
            task3 = progress.add_task(
                f"Upserting {len(raw_df):,} raw rows…", total=None
            )
            raw_records = build_raw_records(raw_df)
            n_raw = db.bulk_upsert_rest(
                "raw_transactions",
                raw_records,
                on_conflict="",
                batch_size=batch_size,
            )
            summary["raw_inserted"] = n_raw
            progress.update(task3, completed=1, total=1)

        # ── Step 4: Upsert cleaned transactions ─────────────
        if not skip_cleaned:
            task4 = progress.add_task(
                f"Upserting {len(cleaned):,} cleaned rows…", total=None
            )
            cleaned_records = build_cleaned_records(cleaned)
            n_cleaned = db.bulk_upsert_rest(
                "transactions",
                cleaned_records,
                on_conflict="",
                batch_size=batch_size,
            )
            summary["cleaned_inserted"] = n_cleaned
            progress.update(task4, completed=1, total=1)
        else:
            n_cleaned = len(cleaned) # for the pipeline log

        # ── Step 5: Seed customers table ─────────────────────
        n_customers = 0
        if not skip_customers:
            task5 = progress.add_task("Seeding customers…", total=None)
            customers_df = build_customers_from_transactions(cleaned)
            customer_records = customers_df.with_columns(
                pl.col("first_purchase_date").dt.to_string("%Y-%m-%dT%H:%M:%S+00:00"),
                pl.col("last_purchase_date").dt.to_string("%Y-%m-%dT%H:%M:%S+00:00"),
                pl.col("total_revenue").cast(pl.Float64),
            ).to_dicts()

            n_customers = db.bulk_upsert_rest(
                "customers",
                customer_records,
                on_conflict="customer_id",
                batch_size=batch_size,
            )
            summary["customers_inserted"] = n_customers
            progress.update(task5, completed=1, total=1)

        # ── Step 6: Log pipeline run ─────────────────────────
        finished = datetime.now(timezone.utc)
        run_record = {
            "run_id": run_id,
            "pipeline_name": "data_ingestion",
            "status": "success",
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "records_processed": n_cleaned,
            "metadata": {
                "csv_path": str(csv_path),
                "raw_rows": len(raw_df),
                "cleaned_rows": len(cleaned),
                "customers": n_customers,
            },
        }
        db.bulk_upsert_rest("pipeline_runs", [run_record], on_conflict="run_id")

    # ── Step 7: Automatic Verification ──────────────────
    task7 = progress.add_task("Verifying database parity…", total=None)
    db_counts = db.execute_sql("""
        SELECT 
            (SELECT COUNT(*) FROM raw_transactions) AS raw_count,
            (SELECT COUNT(*) FROM transactions)     AS cleaned_count,
            (SELECT COUNT(*) FROM customers)        AS customer_count
    """)[0]
    
    summary["verification"] = {
        "raw_parity": db_counts["raw_count"] >= summary.get("raw_rows", 0),
        "cleaned_parity": db_counts["cleaned_count"] >= summary.get("cleaned_rows", 0),
        "db_counts": db_counts
    }
    progress.update(task7, completed=1, total=1)

    summary["duration_seconds"] = (finished - started).total_seconds()
    
    # ── Final Report ────────────────────────────────────
    console.print(f"\n[green]✓ Ingestion complete[/green] — run_id: {run_id}")
    if not summary["verification"]["cleaned_parity"]:
        console.print("[bold red]⚠ WARNING: Database count mismatch detected![/bold red]")
    else:
        console.print("[bold green]✓ Database parity verified.[/bold green]")
        
    console.print_json(data=summary)
    return summary



# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

@app.command()
def ingest(
    csv_path: Path = typer.Argument(
        ...,
        help="Path to the UCI Online Retail CSV/XLSX file",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write to DB"),
    truncate: bool = typer.Option(False, "--truncate", help="Clear tables before ingestion"),
    skip_raw: bool = typer.Option(False, "--skip-raw", help="Skip raw table load"),
    skip_cleaned: bool = typer.Option(False, "--skip-cleaned", help="Skip cleaned table load"),
    skip_customers: bool = typer.Option(False, "--skip-customers", help="Skip customers table load"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Batch size for upsert"),
) -> None:
    """Load UCI Online Retail dataset into Supabase."""
    run_ingestion(
        csv_path, 
        dry_run=dry_run,
        truncate=truncate,
        skip_raw=skip_raw,
        skip_cleaned=skip_cleaned,
        skip_customers=skip_customers,
        batch_size=batch_size
    )


if __name__ == "__main__":
    app()