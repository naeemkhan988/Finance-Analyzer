"""
Query Routes
============
API endpoints for document querying and chat.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Query request body."""
    question: str
    session_id: str = "default"
    k: int = 5
    filter_source: Optional[str] = None
    use_hybrid: bool = True


class QueryResponse(BaseModel):
    """Query response body."""
    success: bool
    answer: str = ""
    sources: list = []
    metadata: dict = {}


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, body: QueryRequest):
    """Query documents using the RAG pipeline."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if not pipeline:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please upload documents first.",
        )

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = pipeline.query(
            question=body.question,
            session_id=body.session_id,
            k=body.k,
            filter_source=body.filter_source,
            use_hybrid=body.use_hybrid,
        )

        return QueryResponse(
            success=True,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(
            success=False,
            answer=f"Error processing query: {str(e)}",
        )


@router.get("/history/{session_id}")
async def get_history(session_id: str, request: Request):
    """Get conversation history for a session."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if not pipeline:
        return {"history": []}

    history = pipeline.get_conversation_history(session_id)
    return {"session_id": session_id, "history": history}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str, request: Request):
    """Clear conversation history for a session."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if pipeline:
        pipeline.clear_conversation(session_id)

    return {"success": True, "message": f"Session '{session_id}' cleared"}
