# Health and analytics routes — health check, system stats, and analytics API endpoints.
"""
Health Routes
=============
Health check, system status, and analytics endpoints.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta

from fastapi import APIRouter, Request, Query

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


@router.get("/analytics")
async def get_analytics(request: Request, days: int = Query(default=7, ge=1, le=90)):
    """
    Get comprehensive analytics data for the dashboard.

    Args:
        days: Number of days to look back (default 7, max 90)
    """
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    result = {
        "query_volume": [],
        "avg_response_time": [],
        "top_questions": [],
        "cache_stats": {},
        "document_stats": {},
    }

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    # Query conversation database
    try:
        conv_db_path = os.path.join("./data", "conversations.db")
        if os.path.exists(conv_db_path):
            conn = sqlite3.connect(conv_db_path)
            conn.row_factory = sqlite3.Row

            # Queries per day
            rows = conn.execute(
                """
                SELECT DATE(timestamp) as day, COUNT(*) as count
                FROM conversations
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY day
                """,
                (cutoff,),
            ).fetchall()
            result["query_volume"] = [
                {"date": row["day"], "count": row["count"]} for row in rows
            ]

            # Average response time per day (from metadata JSON)
            rows = conn.execute(
                """
                SELECT DATE(timestamp) as day, metadata
                FROM conversations
                WHERE timestamp >= ?
                """,
                (cutoff,),
            ).fetchall()

            day_times = {}
            for row in rows:
                day = row["day"]
                try:
                    meta = json.loads(row["metadata"]) if row["metadata"] else {}
                    qt = meta.get("query_time_seconds", 0)
                    if qt > 0:
                        if day not in day_times:
                            day_times[day] = []
                        day_times[day].append(qt)
                except (json.JSONDecodeError, TypeError):
                    pass

            result["avg_response_time"] = [
                {
                    "date": day,
                    "avg_seconds": round(sum(times) / len(times), 2),
                }
                for day, times in sorted(day_times.items())
            ]

            # Top 10 most frequent questions
            rows = conn.execute(
                """
                SELECT question, COUNT(*) as frequency
                FROM conversations
                WHERE timestamp >= ?
                GROUP BY question
                ORDER BY frequency DESC
                LIMIT 10
                """,
                (cutoff,),
            ).fetchall()
            result["top_questions"] = [
                {"question": row["question"], "frequency": row["frequency"]}
                for row in rows
            ]

            conn.close()
    except Exception as e:
        logger.warning(f"Analytics query failed: {e}")

    # Cache stats
    if pipeline and hasattr(pipeline, "semantic_cache"):
        try:
            result["cache_stats"] = pipeline.semantic_cache.get_stats()
        except Exception:
            pass

    # Document stats
    if pipeline and hasattr(pipeline, "document_registry"):
        try:
            result["document_stats"] = pipeline.document_registry.get_stats()
            result["documents"] = pipeline.document_registry.list_documents()
        except Exception:
            pass

    return result
