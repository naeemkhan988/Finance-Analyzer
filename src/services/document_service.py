"""
Document Service
================
Business logic for document upload, processing, and management.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.utils.helpers import allowed_file, sanitize_filename, generate_file_id

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service layer for document management operations.

    Handles:
    - File upload validation and storage
    - Document ingestion into RAG pipeline
    - Document listing and deletion
    - Processing status tracking
    """

    def __init__(
        self,
        upload_folder: str = "./data/raw",
        allowed_extensions: Optional[set] = None,
        max_file_size_mb: int = 50,
    ):
        self.upload_folder = Path(upload_folder)
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.allowed_extensions = allowed_extensions or {"pdf", "docx", "txt", "xlsx", "csv"}
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._processing_status: Dict[str, Dict] = {}

    def validate_file(self, filename: str, file_size: int) -> Dict:
        """
        Validate uploaded file.

        Returns:
            Dict with 'valid' bool and 'error' message if invalid
        """
        if not filename:
            return {"valid": False, "error": "No filename provided"}

        if not allowed_file(filename, self.allowed_extensions):
            return {
                "valid": False,
                "error": f"File type not allowed. Supported: {', '.join(self.allowed_extensions)}",
            }

        if file_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            return {
                "valid": False,
                "error": f"File too large. Maximum size: {max_mb}MB",
            }

        return {"valid": True}

    def save_file(self, file_storage, filename: str) -> Dict:
        """
        Save uploaded file to disk.

        Args:
            file_storage: File-like object (e.g., from request.files)
            filename: Original filename

        Returns:
            Dict with saved file path and metadata
        """
        safe_name = sanitize_filename(filename)
        file_id = generate_file_id(safe_name)
        save_path = self.upload_folder / f"{file_id}_{safe_name}"

        try:
            file_storage.save(str(save_path))

            file_size = os.path.getsize(save_path)

            result = {
                "success": True,
                "file_id": file_id,
                "filename": safe_name,
                "file_path": str(save_path),
                "file_size_bytes": file_size,
                "uploaded_at": datetime.utcnow().isoformat(),
            }

            self._processing_status[file_id] = {
                "status": "uploaded",
                "filename": safe_name,
                "uploaded_at": result["uploaded_at"],
            }

            logger.info(f"File saved: {safe_name} ({file_size} bytes)")
            return result

        except Exception as e:
            logger.error(f"Failed to save file '{filename}': {e}")
            return {"success": False, "error": str(e)}

    def list_uploaded_files(self) -> List[Dict]:
        """List all uploaded files with metadata."""
        files = []

        for path in self.upload_folder.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                files.append({
                    "filename": path.name,
                    "file_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).isoformat(),
                })

        return sorted(files, key=lambda x: x["modified_at"], reverse=True)

    def delete_file(self, filename: str) -> Dict:
        """Delete an uploaded file."""
        file_path = self.upload_folder / filename

        if not file_path.exists():
            return {"success": False, "error": f"File '{filename}' not found"}

        try:
            os.remove(file_path)
            logger.info(f"File deleted: {filename}")
            return {"success": True, "message": f"File '{filename}' deleted"}
        except Exception as e:
            logger.error(f"Failed to delete '{filename}': {e}")
            return {"success": False, "error": str(e)}

    def get_processing_status(self, file_id: str) -> Dict:
        """Get the processing status of a file."""
        return self._processing_status.get(
            file_id, {"status": "unknown", "error": "File ID not found"}
        )

    def update_processing_status(self, file_id: str, status: str, **kwargs):
        """Update processing status for a file."""
        if file_id in self._processing_status:
            self._processing_status[file_id]["status"] = status
            self._processing_status[file_id].update(kwargs)
