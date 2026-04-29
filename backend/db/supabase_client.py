"""
Supabase client + SQLAlchemy engine factory.

Two separate connection strategies:
  1. supabase-py  — used for Supabase Realtime, Auth, Storage, and simple CRUD.
  2. SQLAlchemy   — used for bulk inserts, complex queries, and migrations.
"""

from __future__ import annotations

import os
import json
from functools import lru_cache
from typing import AsyncGenerator, Generator

from loguru import logger
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from supabase import Client, create_client
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from backend.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Supabase-py client  (singleton)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Return a cached Supabase Python client (uses anon key)."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment."
        )
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    logger.info("Supabase client initialised — {}", settings.SUPABASE_URL)
    return client


@lru_cache(maxsize=1)
def get_supabase_admin_client() -> Client:
    """Return a cached Supabase client using the service-role key (bypasses RLS)."""
    if not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise EnvironmentError("SUPABASE_SERVICE_ROLE_KEY must be set in environment.")
    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    logger.info("Supabase admin client initialised (service-role)")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy  (sync + async)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_db_engine(echo: bool = False) -> Engine:
    """Synchronous SQLAlchemy engine with connection pooling."""
    database_url = settings.DATABASE_URL
    if not database_url:
        raise EnvironmentError("DATABASE_URL must be set in environment.")

    engine = create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=echo,
    )

    # Enable pgvector extension on first connect (idempotent)
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_connection, connection_record):  # noqa: ANN001
        with dbapi_connection.cursor() as cur:
            cur.execute("SET search_path TO public;")

    logger.info("SQLAlchemy sync engine created — pool_size=5, max_overflow=10")
    return engine


@lru_cache(maxsize=1)
def get_async_db_engine(echo: bool = False) -> AsyncEngine:
    """Async SQLAlchemy engine (asyncpg driver)."""
    async_url = settings.DATABASE_URL_ASYNC
    if not async_url:
        # Auto-convert sync URL to async URL
        sync_url = settings.DATABASE_URL or ""
        async_url = sync_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        ).replace("postgres://", "postgresql+asyncpg://")

    engine = create_async_engine(
        async_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=echo,
    )
    logger.info("SQLAlchemy async engine created")
    return engine


def get_session_factory(echo: bool = False) -> sessionmaker:
    """Return a synchronous session factory."""
    return sessionmaker(bind=get_db_engine(echo=echo), autocommit=False, autoflush=False)


def get_async_session_factory(echo: bool = False) -> async_sessionmaker:
    """Return an async session factory."""
    return async_sessionmaker(
        bind=get_async_db_engine(echo=echo),
        class_=AsyncSession,
        expire_on_commit=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context-manager helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db_session() -> Generator[Session, None, None]:
    """Yield a synchronous DB session (use as context manager or FastAPI dep)."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session (FastAPI dependency)."""
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─────────────────────────────────────────────────────────────────────────────
# SupabaseClient — convenience wrapper used throughout the pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SupabaseClient:
    """
    Thin wrapper around both the supabase-py client and SQLAlchemy engine.
    Used by feature pipelines for reads/writes.
    """

    def __init__(self, use_service_role: bool = False) -> None:
        self._sb = (
            get_supabase_admin_client() if use_service_role else get_supabase_client()
        )
        self._engine = get_db_engine()

    # ── supabase-py passthrough ──────────────────────────────

    @property
    def table(self):  # noqa: ANN201
        return self._sb.table

    @property
    def rpc(self):  # noqa: ANN201
        return self._sb.rpc

    # ── Bulk helpers (SQLAlchemy) ────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(lambda exc: not isinstance(exc, IntegrityError)),
        reraise=True,
    )
    def bulk_upsert(
        self,
        table_name: str,
        records: list[dict],
        conflict_columns: list[str] | None = None,
        batch_size: int = 500,
    ) -> int:
        """
        Bulk upsert records into a Supabase table via raw SQL.
        Returns total rows inserted/updated.
        """
        if not records:
            return 0

        total = 0
        normalized_records: list[dict] = []
        for record in records:
            normalized: dict = {}
            for key, value in record.items():
                if isinstance(value, (dict, list)):
                    normalized[key] = json.dumps(value)
                else:
                    normalized[key] = value
            normalized_records.append(normalized)

        columns = list(normalized_records[0].keys())
        col_list = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(f":{c}" for c in columns)

        conflict_clause = ""
        if conflict_columns:
            conflict_target = ", ".join(f'"{c}"' for c in conflict_columns)
            updates = ", ".join(
                f'"{c}" = EXCLUDED."{c}"'
                for c in columns
                if c not in conflict_columns
            )
            if updates:
                conflict_clause = (
                    f"ON CONFLICT ({conflict_target}) DO UPDATE SET {updates}"
                )
            else:
                conflict_clause = f"ON CONFLICT ({conflict_target}) DO NOTHING"
        else:
            conflict_clause = "ON CONFLICT DO NOTHING"

        sql = text(
            f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders}) {conflict_clause}'
        )

        with self._engine.begin() as conn:
            for i in range(0, len(normalized_records), batch_size):
                batch = normalized_records[i : i + batch_size]
                conn.execute(sql, batch)
                total += len(batch)
                logger.debug(
                    "Upserted batch {}/{} into {} ({} rows)",
                    i // batch_size + 1,
                    (len(normalized_records) - 1) // batch_size + 1,
                    table_name,
                    len(batch),
                )

        logger.info("bulk_upsert → {} rows into {}", total, table_name)
        return total

    def bulk_upsert_rest(
        self,
        table_name: str,
        records: list[dict],
        on_conflict: str = "",
        batch_size: int = 1000,
    ) -> int:
        """
        Bulk upsert via Supabase REST API (HTTPS port 443).
        Faster and more reliable than direct postgres for one-time loads.
        on_conflict: comma-separated column names for conflict resolution.
        """
        if not records:
            return 0

        total = 0
        n_batches = (len(records) - 1) // batch_size + 1
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            kwargs = {"returning": "minimal"}
            if on_conflict:
                kwargs["on_conflict"] = on_conflict
                kwargs["ignore_duplicates"] = True
            self._sb.table(table_name).upsert(batch, **kwargs).execute()
            total += len(batch)
            logger.debug(
                "REST upsert batch {}/{} into {} ({} rows)",
                i // batch_size + 1,
                n_batches,
                table_name,
                len(batch),
            )
        logger.info("bulk_upsert_rest → {} rows into {}", total, table_name)
        return total

    def execute_sql(self, sql: str, params: dict | None = None) -> list[dict]:
        """Execute arbitrary SQL and return rows as dicts."""
        # Use .begin() for automatic commit/rollback
        with self._engine.begin() as conn:
            result = conn.execute(text(sql), params or {})
            if result.returns_rows:
                keys = list(result.keys())
                return [dict(zip(keys, row)) for row in result.fetchall()]
            return []

    def health_check(self) -> bool:
        """Verify database connectivity."""
        try:
            rows = self.execute_sql("SELECT 1 AS ok")
            return rows[0].get("ok") == 1
        except Exception as exc:
            logger.error("Database health check failed: {}", exc)
            return False
