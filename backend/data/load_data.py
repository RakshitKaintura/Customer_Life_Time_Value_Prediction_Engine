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
        df = pl.read_csv(
            path,
            schema_overrides={k: v for k, v in SCHEMA_OVERRIDES.items()},
            try_parse_dates=True,
            null_values=["", "NA", "N/A", "NULL", "null"],
            truncate_ragged_lines=True,
            ignore_errors=True,
        )

    # Rename to snake_case
    existing_renames = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(existing_renames)

    # Parse InvoiceDate if it came in as string
    if df["invoice_date"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("invoice_date")
                .str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False)
                .alias("invoice_date")
        )

    # Ensure timezone-aware
    if df["invoice_date"].dtype == pl.Datetime:
        df = df.with_columns(
            pl.col("invoice_date")
                .dt.replace_time_zone("UTC")
        )

    # Normalise customer_id to string
    df = df.with_columns(
        pl.col("customer_id")
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.replace(r"\.0$", "")   # remove ".0" from float-cast IDs
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
        df.with_columns(
            pl.col("invoice_date").dt.to_string("%Y-%m-%dT%H:%M:%S+00:00"),
            pl.col("customer_id").cast(pl.Utf8),
        )
        .to_dicts()
    )


def build_cleaned_records(df: pl.DataFrame) -> list[dict]:
    """Convert cleaned DataFrame to list of dicts for upsert."""
    return (
        df.with_columns(
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
    skip_raw: bool = False,
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

        # ── Step 3: Upsert raw transactions ─────────────────
        if not skip_raw:
            task3 = progress.add_task(
                f"Upserting {len(raw_df):,} raw rows…", total=None
            )
            raw_records = build_raw_records(raw_df)
            n_raw = db.bulk_upsert(
                "raw_transactions",
                raw_records,
                conflict_columns=None,
                batch_size=batch_size,
            )
            summary["raw_inserted"] = n_raw
            progress.update(task3, completed=1, total=1)

        # ── Step 4: Upsert cleaned transactions ─────────────
        task4 = progress.add_task(
            f"Upserting {len(cleaned):,} cleaned rows…", total=None
        )
        cleaned_records = build_cleaned_records(cleaned)
        n_cleaned = db.bulk_upsert(
            "transactions",
            cleaned_records,
            conflict_columns=None,
            batch_size=batch_size,
        )
        summary["cleaned_inserted"] = n_cleaned
        progress.update(task4, completed=1, total=1)

        # ── Step 5: Seed customers table ─────────────────────
        task5 = progress.add_task("Seeding customers…", total=None)
        customers_df = build_customers_from_transactions(cleaned)
        customer_records = customers_df.with_columns(
            pl.col("first_purchase_date").dt.to_string("%Y-%m-%dT%H:%M:%S+00:00"),
            pl.col("last_purchase_date").dt.to_string("%Y-%m-%dT%H:%M:%S+00:00"),
            pl.col("total_revenue").cast(pl.Float64),
        ).to_dicts()

        n_customers = db.bulk_upsert(
            "customers",
            customer_records,
            conflict_columns=["customer_id"],
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
        db.bulk_upsert("pipeline_runs", [run_record], conflict_columns=["run_id"])

    summary["duration_seconds"] = (finished - started).total_seconds()
    console.print(f"\n[green]✓ Ingestion complete[/green] — run_id: {run_id}")
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
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse but don't write to DB"),
    skip_raw: bool = typer.Option(False, "--skip-raw", help="Skip raw_transactions upsert"),
    batch_size: int = typer.Option(500, "--batch-size", help="Upsert batch size"),
) -> None:
    """Load UCI Online Retail dataset into Supabase."""
    run_ingestion(csv_path, dry_run=dry_run, skip_raw=skip_raw, batch_size=batch_size)


if __name__ == "__main__":
    app()