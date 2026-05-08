# Upload routes — secure document upload and management API endpoints.
"""
Upload Routes
=============
API endpoints for document upload and management with path traversal protection.
"""

import os
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename

from backend.schemas.request import UploadResponse
from backend.schemas.response import DocumentListResponse

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path(os.getenv("UPLOAD_FOLDER", "./data/raw")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "xlsx", "csv"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", 50)) * 1024 * 1024


def _validate_file_path(filename: str) -> Path:
    """
    Validate and secure a filename against path traversal attacks.
    Returns the resolved safe file path within UPLOAD_DIR.
    Raises HTTPException 400 if path traversal detected.
    """
    safe_name = secure_filename(filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename after sanitization")

    file_path = (UPLOAD_DIR / safe_name).resolve()

    # Ensure resolved path stays within UPLOAD_DIR
    if not str(file_path).startswith(str(UPLOAD_DIR)):
        logger.warning(
            f"Path traversal attempt blocked: original='{filename}', "
            f"resolved='{file_path}', upload_dir='{UPLOAD_DIR}'"
        )
        raise HTTPException(
            status_code=400,
            detail="Path traversal detected. Filename rejected.",
        )

    return file_path


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload and ingest a financial document."""
    client_ip = request.client.host if request.client else "unknown"

    # Validate filename exists
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    logger.info(f"Upload attempt: filename='{file.filename}', client_ip={client_ip}")

    # Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(
            f"Upload rejected: invalid extension '.{ext}', "
            f"filename='{file.filename}', client_ip={client_ip}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' not allowed. Supported: {ALLOWED_EXTENSIONS}",
        )

    # Secure filename and validate path
    file_path = _validate_file_path(file.filename)

    # Read and validate file size
    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB",
            )

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(
            f"File saved: '{file_path.name}' ({len(contents)} bytes), client_ip={client_ip}"
        )

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
                filename=file_path.name,
                message=f"Document ingested: {result.get('chunks_created', 0)} chunks created",
                chunks_created=result.get("chunks_created", 0),
                processing_time=result.get("processing_time_seconds", 0),
            )
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return UploadResponse(
                success=False,
                filename=file_path.name,
                message=f"File saved but ingestion failed: {e}",
            )
    else:
        return UploadResponse(
            success=True,
            filename=file_path.name,
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
    # Secure the filename
    safe_name = secure_filename(filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    pipeline = getattr(request.app.state, "rag_pipeline", None)

    # Delete from pipeline
    if pipeline:
        result = pipeline.delete_document(safe_name)
    else:
        result = {"success": True}

    # Delete file from disk
    file_path = (UPLOAD_DIR / safe_name).resolve()
    if str(file_path).startswith(str(UPLOAD_DIR)) and file_path.exists():
        os.remove(file_path)

    return result


@router.get("/documents/{filename}/preview")
async def preview_document(filename: str, request: Request):
    """Preview first 3 pages of a document with summary."""
    safe_name = secure_filename(filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = (UPLOAD_DIR / safe_name).resolve()
    if not str(file_path).startswith(str(UPLOAD_DIR)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        # Load first 3 pages only
        all_pages = pipeline.doc_processor.load_document(str(file_path))
        preview_pages = all_pages[:3]

        summary = pipeline.doc_processor.get_document_summary(all_pages) if all_pages else {}

        preview_data = []
        for page in preview_pages:
            preview_data.append({
                "page": page["metadata"].get("page", 1),
                "content_preview": page["content"][:1000],
            })

        # Detect entities from preview
        all_text = " ".join(p["content"] for p in preview_pages)
        detected_entities = pipeline.doc_processor._detect_financial_entities(all_text)

        return {
            "filename": safe_name,
            "summary": summary,
            "preview_pages": preview_data,
            "detected_entities": {k: len(v) for k, v in detected_entities.items()},
            "table_count": summary.get("total_tables", 0),
            "has_financial_data": summary.get("has_financial_data", False),
        }
    except Exception as e:
        logger.error(f"Preview failed for '{safe_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {e}")
