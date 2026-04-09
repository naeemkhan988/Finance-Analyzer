"""
Response Schemas
================
Pydantic models for standardized API responses.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class SourceInfo(BaseModel):
    """Source citation information."""
    document: str = ""
    page: Any = ""
    relevance_score: float = 0.0
    has_tables: bool = False
    chunk_preview: str = ""


class QueryResponseSchema(BaseModel):
    """Response for document query."""
    success: bool
    answer: str = ""
    sources: List[SourceInfo] = []
    metadata: Dict = {}


class DocumentInfo(BaseModel):
    """Document information."""
    source: str = ""
    total_pages: int = 0
    chunks: int = 0
    has_tables: bool = False
    type: str = ""


class DocumentListResponse(BaseModel):
    """Response for document listing."""
    documents: List[DocumentInfo] = []
    total: int = 0


class StatsResponse(BaseModel):
    """System statistics response."""
    status: str = "unknown"
    pipeline_active: bool = False
    engine: Dict = {}
    vector_store: Dict = {}
    llm: Dict = {}
