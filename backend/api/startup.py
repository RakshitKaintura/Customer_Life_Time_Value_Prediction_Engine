"""
FastAPI lifespan startup/shutdown.

Loads all models into memory on startup so they are ready
for the first request with zero cold-start delay.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from backend.api.config import api_settings
from backend.api.dependencies import (
    set_cold_start_scorer,
    set_scoring_engine,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    All startup logic runs before yield; shutdown after.
    """
    logger.info("=== LTV Scoring API starting up ===")
    logger.info("Environment: {}", api_settings.ENVIRONMENT)

    await _load_models()

    logger.info("=== Startup complete. Ready to serve. ===")
    yield

    # Shutdown
    logger.info("=== LTV Scoring API shutting down ===")


async def _load_models() -> None:
    """Load all ML models and initialise the scoring engine."""
    import pickle
    from backend.db.supabase_client import SupabaseClient
    from backend.ml.cold_start import ColdStartScorer

    db = SupabaseClient(use_service_role=True)

    # ── 1. Cold-start scorer (always available) ──────────────
    logger.info("Loading cold-start scorer...")
    cold_scorer = ColdStartScorer(db)
    try:
        cold_scorer.load_table()
    except Exception as exc:
        logger.warning("Cold-start table load failed: {} — using empty table", exc)
    set_cold_start_scorer(cold_scorer)
    logger.info("Cold-start scorer loaded")

    # ── 2. BG/NBD model ──────────────────────────────────────
    bgnbd_model = None
    models_dir  = api_settings.MODELS_DIR
    bgnbd_meta_path = models_dir / f"{api_settings.BGNBD_MODEL_VERSION}_meta.pkl"

    if bgnbd_meta_path.exists():
        try:
            from backend.ml.bgnbd_model import BGNBDModel
            bgnbd_model = BGNBDModel.load_from_disk(
                models_dir, api_settings.BGNBD_MODEL_VERSION
            )
            logger.info("BG/NBD model loaded: {}", api_settings.BGNBD_MODEL_VERSION)
        except Exception as exc:
            logger.warning("BG/NBD load failed: {}", exc)
    else:
        logger.warning("BG/NBD model not found at {} — some scoring paths unavailable", bgnbd_meta_path)

    # ── 3. ONNX Runtime for Transformer ──────────────────────
    onnx_engine = None
    onnx_path   = api_settings.ONNX_PATH

    if Path(onnx_path).exists():
        try:
            from backend.ml.transformer_onnx import ONNXInferenceEngine
            onnx_engine = ONNXInferenceEngine(str(onnx_path))
            onnx_engine.warmup(max_seq_len=api_settings.MAX_SEQ_LEN)
            logger.info("ONNX Runtime loaded and warmed up: {}", onnx_path)
        except Exception as exc:
            logger.warning("ONNX load failed: {}", exc)
    else:
        logger.warning("ONNX model not found at {}", onnx_path)

    # ── 4. XGBoost Fusion model ───────────────────────────────
    fusion_model = None
    fusion_meta  = models_dir / f"{api_settings.FUSION_MODEL_VERSION}_meta.pkl"

    if fusion_meta.exists():
        try:
            from backend.ml.fusion import XGBoostMetaLearner
            fusion_model = XGBoostMetaLearner.load_from_disk(
                models_dir, api_settings.FUSION_MODEL_VERSION
            )
            logger.info("Fusion model loaded: {}", api_settings.FUSION_MODEL_VERSION)
        except Exception as exc:
            logger.warning("Fusion model load failed: {}", exc)
    else:
        logger.warning("Fusion model not found at {}", fusion_meta)

    # ── 5. Assemble scoring engine ────────────────────────────
    if bgnbd_model and onnx_engine and fusion_model:
        from backend.ml.scoring_engine import LTVScoringEngine
        engine = LTVScoringEngine(
            bgnbd_model        = bgnbd_model,
            onnx_engine        = onnx_engine,
            fusion_model       = fusion_model,
            cold_start_scorer  = cold_scorer,
            db_client          = db,
            max_seq_len        = api_settings.MAX_SEQ_LEN,
            model_version      = api_settings.FUSION_MODEL_VERSION,
        )
        set_scoring_engine(engine)
        logger.info("Full scoring engine assembled and registered")
    else:
        logger.warning(
            "Scoring engine NOT assembled — missing: bgnbd={} onnx={} fusion={}",
            bgnbd_model is not None,
            onnx_engine is not None,
            fusion_model is not None,
        )
        # Register a stub so the health endpoint doesn't crash
        set_scoring_engine(None)