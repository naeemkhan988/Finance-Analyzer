"""
Helper Utilities
================
Common helper functions used across the application.
"""

import os
import uuid
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:8]


def generate_file_id(filename: str) -> str:
    """Generate a unique file ID based on filename and timestamp."""
    timestamp = datetime.utcnow().isoformat()
    raw = f"{filename}_{timestamp}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def allowed_file(filename: str, allowed_extensions: set = None) -> bool:
    """Check if a file extension is allowed."""
    if allowed_extensions is None:
        allowed_extensions = {"pdf", "docx", "txt", "xlsx", "csv"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage."""
    sanitized = filename.replace("/", "_").replace("\\", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in "._-")
    return sanitized or "unnamed_file"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to human-readable format."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%b %d, %Y %I:%M %p")
    except (ValueError, TypeError):
        return iso_timestamp
