"""Feature engineering modules — Polars + DuckDB."""

from backend.features.rfm import RFMPipeline
from backend.features.cohorts import CohortPipeline
from backend.features.duckdb_agg import DuckDBAggregator
from backend.features.sequences import SequenceBuilder

__all__ = ["RFMPipeline", "CohortPipeline", "DuckDBAggregator", "SequenceBuilder"]