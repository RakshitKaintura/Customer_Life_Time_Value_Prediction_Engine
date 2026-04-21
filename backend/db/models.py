"""
SQLAlchemy 2.0 ORM models + Pydantic v2 schemas for the LTV Prediction Engine.

ORM models  → used for direct DB operations via SQLAlchemy sessions.
Pydantic schemas → used for API validation and inter-service data contracts.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ─────────────────────────────────────────────────────────────────────────────
# ORM Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class RawTransaction(Base):
    __tablename__ = "raw_transactions"

    id: Mapped[int]              = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    invoice_no: Mapped[str]      = mapped_column(String(20), nullable=False)
    stock_code: Mapped[str]      = mapped_column(String(20), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    quantity: Mapped[int]        = mapped_column(Integer, nullable=False)
    invoice_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    unit_price: Mapped[Decimal]  = mapped_column(Numeric(10, 4), nullable=False)
    customer_id: Mapped[str | None] = mapped_column(String(20))
    country: Mapped[str | None]  = mapped_column(String(50))
    source_dataset: Mapped[str]  = mapped_column(String(50), default="uci_online_retail")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int]              = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    invoice_no: Mapped[str]      = mapped_column(String(20), nullable=False)
    stock_code: Mapped[str]      = mapped_column(String(20), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    quantity: Mapped[int]        = mapped_column(Integer, nullable=False)
    invoice_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    unit_price: Mapped[Decimal]  = mapped_column(Numeric(10, 4), nullable=False)
    customer_id: Mapped[str]     = mapped_column(String(20), nullable=False)
    country: Mapped[str | None]  = mapped_column(String(50))
    product_category: Mapped[str | None] = mapped_column(String(80))
    amount_bucket: Mapped[int | None]    = mapped_column(SmallInteger)
    is_returned: Mapped[bool]    = mapped_column(Boolean, default=False)
    source_dataset: Mapped[str]  = mapped_column(String(50), default="uci_online_retail")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str]     = mapped_column(String(20), primary_key=True)
    country: Mapped[str | None]  = mapped_column(String(50))
    first_purchase_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_purchase_date: Mapped[datetime | None]  = mapped_column(DateTime(timezone=True))
    total_orders: Mapped[int]    = mapped_column(Integer, default=0)
    total_revenue: Mapped[Decimal] = mapped_column(Numeric(14, 4), default=0)
    acquisition_channel: Mapped[str] = mapped_column(String(50), default="organic")
    vertical: Mapped[str | None] = mapped_column(String(80))
    company_size: Mapped[str | None] = mapped_column(String(30))
    plan_tier: Mapped[str | None]    = mapped_column(String(30))
    is_synthetic: Mapped[bool]   = mapped_column(Boolean, default=False)
    source_dataset: Mapped[str]  = mapped_column(String(50), default="uci_online_retail")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    rfm_features: Mapped[list["RFMFeatures"]] = relationship(
        back_populates="customer", lazy="select"
    )
    ltv_predictions: Mapped[list["LTVPrediction"]] = relationship(
        back_populates="customer", lazy="select"
    )


class RFMFeatures(Base):
    __tablename__ = "rfm_features"

    id: Mapped[int]              = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str]     = mapped_column(
        String(20), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    observation_end_date: Mapped[date] = mapped_column(Date, nullable=False)

    # BG/NBD inputs
    recency_days: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    frequency: Mapped[int | None]        = mapped_column(Integer)
    monetary_avg: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    t_days: Mapped[int | None]           = mapped_column(Integer)

    # Extended monetary
    monetary_total: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    monetary_std: Mapped[Decimal | None]   = mapped_column(Numeric(12, 4))
    purchase_variance: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))

    # Purchase behaviour
    orders_count: Mapped[int | None]            = mapped_column(Integer)
    unique_products: Mapped[int | None]         = mapped_column(Integer)
    unique_categories: Mapped[int | None]       = mapped_column(Integer)
    unique_invoices: Mapped[int | None]         = mapped_column(Integer)
    avg_items_per_order: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    avg_days_between_orders: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    std_days_between_orders: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))

    # Cohort signals
    cohort_month: Mapped[str | None]              = mapped_column(String(7))
    days_to_second_purchase: Mapped[int | None]   = mapped_column(Integer)
    first_purchase_category: Mapped[str | None]   = mapped_column(String(80))
    first_purchase_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    # Flags
    has_returned_items: Mapped[bool] = mapped_column(Boolean, default=False)
    multi_country: Mapped[bool]      = mapped_column(Boolean, default=False)

    # Labels
    actual_ltv_12m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    actual_ltv_24m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    actual_ltv_36m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))

    pipeline_run_id: Mapped[str | None] = mapped_column(String(50))

    # Relationship
    customer: Mapped["Customer"] = relationship(back_populates="rfm_features")


class PurchaseSequence(Base):
    __tablename__ = "purchase_sequences"

    id: Mapped[int]              = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str]     = mapped_column(
        String(20), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False
    )
    sequence_json: Mapped[dict]  = mapped_column(JSONB, nullable=False)
    sequence_length: Mapped[int] = mapped_column(Integer, nullable=False)
    observation_end_date: Mapped[date | None] = mapped_column(Date)
    built_at: Mapped[datetime]   = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CustomerEmbedding(Base):
    __tablename__ = "customer_embeddings"

    id: Mapped[int]              = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str]     = mapped_column(
        String(20), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(64))
    model_version: Mapped[str]   = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LTVPrediction(Base):
    __tablename__ = "ltv_predictions"

    id: Mapped[int]              = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str]     = mapped_column(
        String(20), ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False
    )
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    model_version: Mapped[str]   = mapped_column(String(20), nullable=False)
    ltv_source: Mapped[str]      = mapped_column(String(30), default="full_model")

    ltv_12m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    ltv_24m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    ltv_36m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))

    bgnbd_ltv_12m: Mapped[Decimal | None]          = mapped_column(Numeric(14, 4))
    bgnbd_ltv_36m: Mapped[Decimal | None]          = mapped_column(Numeric(14, 4))
    probability_alive_12m: Mapped[Decimal | None]  = mapped_column(Numeric(6, 4))
    expected_purchases_12m: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))

    transformer_ltv_12m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    transformer_ltv_36m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))

    ci_lower_36m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    ci_upper_36m: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))

    ltv_percentile: Mapped[int | None]     = mapped_column(SmallInteger)
    segment: Mapped[str | None]            = mapped_column(String(20))
    recommended_max_cac: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    top_drivers: Mapped[dict | None]   = mapped_column(JSONB)
    causal_levers: Mapped[dict | None] = mapped_column(JSONB)
    shap_values: Mapped[dict | None]   = mapped_column(JSONB)

    scoring_latency_ms: Mapped[int | None] = mapped_column(Integer)
    pipeline_run_id: Mapped[str | None]    = mapped_column(String(50))

    # Relationship
    customer: Mapped["Customer"] = relationship(back_populates="ltv_predictions")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int]           = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str]       = mapped_column(String(50), unique=True, default=lambda: str(uuid.uuid4()))
    pipeline_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str]       = mapped_column(String(20), default="running")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    records_processed: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None]     = mapped_column(Text)
    run_metadata: Mapped[dict | None]         = mapped_column("metadata", JSONB)
    wandb_run_id: Mapped[str | None]      = mapped_column(String(50))


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic v2 Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TransactionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    invoice_no: str
    stock_code: str
    description: str | None = None
    quantity: int
    invoice_date: datetime
    unit_price: float
    customer_id: str
    country: str | None = None
    product_category: str | None = None
    amount_bucket: int | None = None
    is_returned: bool = False
    source_dataset: str = "uci_online_retail"


class CustomerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customer_id: str
    country: str | None = None
    first_purchase_date: datetime | None = None
    last_purchase_date: datetime | None = None
    total_orders: int = 0
    total_revenue: float = 0.0
    acquisition_channel: str = "organic"
    vertical: str | None = None
    company_size: str | None = None
    plan_tier: str | None = None


class RFMFeaturesSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customer_id: str
    observation_end_date: date
    recency_days: float | None = None
    frequency: int | None = None
    monetary_avg: float | None = None
    t_days: int | None = None
    monetary_total: float | None = None
    monetary_std: float | None = None
    purchase_variance: float | None = None
    orders_count: int | None = None
    unique_products: int | None = None
    unique_categories: int | None = None
    cohort_month: str | None = None
    days_to_second_purchase: int | None = None
    first_purchase_category: str | None = None
    avg_days_between_orders: float | None = None
    actual_ltv_12m: float | None = None
    actual_ltv_36m: float | None = None


class LTVPredictionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    customer_id: str
    ltv_source: str = "full_model"
    ltv_12m: float | None = None
    ltv_24m: float | None = None
    ltv_36m: float | None = None
    ltv_percentile: int | None = None
    segment: str | None = None
    probability_alive_12m: float | None = None
    recommended_max_cac: float | None = None
    confidence_interval_36m: tuple[float, float] | None = None
    top_drivers: list[str] = Field(default_factory=list)
    causal_levers: list[str] = Field(default_factory=list)
    lookalike_customer_ids: list[str] = Field(default_factory=list)
    scoring_latency_ms: int | None = None

    @model_validator(mode="after")
    def build_confidence_interval(self) -> "LTVPredictionSchema":
        # This would be populated from the ORM ci_lower/ci_upper fields
        return self