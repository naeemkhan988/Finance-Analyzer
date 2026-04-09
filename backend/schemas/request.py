"""
Request Schemas
===============
Pydantic models for API request/response validation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response for document upload."""
    success: bool
    filename: str = ""
    message: str = ""
    chunks_created: int = 0
    processing_time: float = 0.0


class QueryRequestSchema(BaseModel):
    """Request body for document query."""
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=50)
    k: int = Field(default=5, ge=1, le=20)
    filter_source: Optional[str] = None
    use_hybrid: bool = True
