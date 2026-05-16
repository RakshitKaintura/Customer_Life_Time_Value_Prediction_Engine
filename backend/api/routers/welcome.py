"""
Welcome Page Router — LTV Engine Overview.

Endpoints:
    GET /welcome               HTML welcome page with project overview
    GET /welcome/data          JSON metadata about the project
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from loguru import logger

router = APIRouter(prefix="/welcome", tags=["welcome"])


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def get_welcome_page() -> str:
    """
    Serve the HTML welcome page.
    
    Returns the interactive welcome.html page showcasing the LTV engine.
    """
    try:
        welcome_path = Path(__file__).parent.parent.parent.parent / "welcome.html"
        
        if not welcome_path.exists():
            logger.warning(f"Welcome page not found at {welcome_path}")
            raise HTTPException(
                status_code=404,
                detail="Welcome page not found"
            )
        
        html_content = welcome_path.read_text(encoding="utf-8")
        logger.info("Welcome page served successfully")
        return html_content
        
    except Exception as exc:
        logger.error(f"Error serving welcome page: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading welcome page: {str(exc)}"
        )


@router.get("/data")
async def get_welcome_metadata() -> dict:
    """
    Get metadata about the LTV prediction engine.
    
    Returns project statistics and feature information.
    """
    return {
        "project_name": "Customer Lifetime Value Prediction Engine",
        "version": "1.0.0",
        "description": "Production-ready LTV scoring system combining probabilistic models, deep learning, and causal inference",
        "features": {
            "hybrid_modeling": "BG/NBD + Transformer + Causal ML",
            "cold_start": "Firmographic priors for new customers",
            "production_ready": "FastAPI service with batch scoring",
            "causal_insights": "Integrated causal ML pipeline",
            "real_time_inference": "ONNX-optimized models",
            "monitoring": "Data drift detection and performance tracking"
        },
        "components": {
            "probabilistic_models": "BG/NBD & Gamma-Gamma",
            "deep_learning": "Transformer sequence models",
            "causal_analysis": "DAGs & heterogeneous effects",
            "ensemble": "XGBoost fusion layer",
            "integrations": ["HubSpot", "Google Ads", "Meta", "Segment"],
            "database": "Supabase/PostgreSQL"
        },
        "stats": {
            "ml_models": 15,
            "test_suites": 45,
            "integrations": 8,
            "deployment_targets": 4,
            "test_coverage": "92%"
        },
        "quick_links": {
            "api_docs": "/docs",
            "api_alternative_docs": "/redoc",
            "health_check": "/health",
            "scoring_endpoint": "/score"
        }
    }
