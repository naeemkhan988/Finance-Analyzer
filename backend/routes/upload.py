"""
Upload Routes
=============
API endpoints for document upload and management.
"""

import os
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.schemas.request import UploadResponse
from backend.schemas.response import DocumentListResponse

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path(os.getenv("UPLOAD_FOLDER", "./data/raw"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "xlsx", "csv"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and ingest a financial document."""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not allowed. Supported: {ALLOWED_EXTENSIONS}",
        )

    # Save file
    file_path = UPLOAD_DIR / file.filename
    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB",
            )

        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"File saved: {file.filename} ({len(contents)} bytes)")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Ingest into RAG pipeline
    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline:
        try:
            result = pipeline.ingest_document(str(file_path))
            return UploadResponse(
                success=result["success"],
                filename=file.filename,
                message=f"Document ingested: {result.get('chunks_created', 0)} chunks created",
                chunks_created=result.get("chunks_created", 0),
                processing_time=result.get("processing_time_seconds", 0),
            )
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return UploadResponse(
                success=False,
                filename=file.filename,
                message=f"File saved but ingestion failed: {e}",
            )
    else:
        return UploadResponse(
            success=True,
            filename=file.filename,
            message="File saved. RAG pipeline not initialized.",
        )


@router.get("/documents")
async def list_documents(request: Request):
    """List all uploaded and ingested documents."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if pipeline:
        docs = pipeline.get_documents()
        return {"documents": docs, "total": len(docs)}

    # Fallback: list files from upload directory
    files = []
    for path in UPLOAD_DIR.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            files.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            })

    return {"documents": files, "total": len(files)}


@router.delete("/documents/{filename}")
async def delete_document(filename: str, request: Request):
    """Delete a document from the system."""
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    # Delete from pipeline
    if pipeline:
        result = pipeline.delete_document(filename)
    else:
        result = {"success": True}

    # Delete file
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        os.remove(file_path)

    return result
