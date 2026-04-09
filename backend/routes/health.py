"""
Health Routes
=============
Health check and system status endpoints.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "multimodal-rag-finance-analyzer",
    }


@router.get("/stats")
async def system_stats(request: Request):
    """Get comprehensive system statistics."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if pipeline:
        stats = pipeline.get_stats()
        return {
            "status": "operational",
            "pipeline_active": True,
            **stats,
        }

    return {
        "status": "limited",
        "pipeline_active": False,
        "message": "RAG pipeline not initialized",
    }


@router.get("/providers")
async def llm_providers(request: Request):
    """Get information about configured LLM providers."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if pipeline and hasattr(pipeline, "llm_router"):
        return pipeline.llm_router.get_provider_info()

    return {"fallback": {"name": "No providers configured", "active": True}}
