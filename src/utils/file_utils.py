"""
File Utilities
==============
Common file handling operations for document processing.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def ensure_directory(path: str) -> Path:
    """Create directory if it doesn't exist."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension."""
    return Path(filename).suffix.lower()


def list_files(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    List files in a directory with optional extension filtering.

    Args:
        directory: Directory path to scan
        extensions: Optional list of extensions to filter (e.g., ['.pdf', '.txt'])
        recursive: Whether to search subdirectories

    Returns:
        List of Path objects
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    if recursive:
        files = list(dir_path.rglob("*"))
    else:
        files = list(dir_path.iterdir())

    # Filter to files only
    files = [f for f in files if f.is_file()]

    # Filter by extension
    if extensions:
        ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        files = [f for f in files if f.suffix.lower() in ext_set]

    return sorted(files)


def get_file_info(file_path: str) -> Dict:
    """Get detailed file information."""
    path = Path(file_path)
    if not path.exists():
        return {"exists": False}

    stat = path.stat()
    return {
        "exists": True,
        "name": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def copy_file(source: str, destination: str) -> bool:
    """Copy a file to a new location."""
    try:
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(f"Failed to copy '{source}' to '{destination}': {e}")
        return False


def move_file(source: str, destination: str) -> bool:
    """Move a file to a new location."""
    try:
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)
        return True
    except Exception as e:
        logger.error(f"Failed to move '{source}' to '{destination}': {e}")
        return False


def safe_delete(file_path: str) -> bool:
    """Safely delete a file."""
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete '{file_path}': {e}")
        return False


def get_directory_size(directory: str) -> int:
    """Get total size of all files in a directory (bytes)."""
    total = 0
    for path in Path(directory).rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total
