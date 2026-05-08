# Document registry — SQLite-backed document metadata registry.
"""
Document Registry
=================
SQLite-backed persistent registry for ingested document metadata.
Tracks file info, processing stats, and document lifecycle.
"""

import json
import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentRegistry:
    """
    Persistent document metadata registry using SQLite.

    Features:
    - Document registration with full metadata
    - Soft delete (status flag, not hard delete)
    - File type statistics
    - Processing time tracking
    """

    def __init__(self, db_path: str = "./data/documents.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create documents table if not exists."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    file_hash TEXT DEFAULT '',
                    file_size_bytes INTEGER DEFAULT 0,
                    file_type TEXT DEFAULT '',
                    total_pages INTEGER DEFAULT 0,
                    chunk_count INTEGER DEFAULT 0,
                    has_tables INTEGER DEFAULT 0,
                    has_images INTEGER DEFAULT 0,
                    processing_time_seconds REAL DEFAULT 0,
                    ingested_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            """)
            conn.commit()
        logger.info(f"DocumentRegistry initialized: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def register(self, filename: str, metadata_dict: Dict):
        """
        Register or update a document with its metadata.

        Args:
            filename: Document filename
            metadata_dict: Dict with keys like file_hash, file_size_bytes,
                          file_type, total_pages, chunk_count, etc.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (filename, file_hash, file_size_bytes, file_type,
                     total_pages, chunk_count, has_tables, has_images,
                     processing_time_seconds, ingested_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    filename,
                    metadata_dict.get("file_hash", ""),
                    metadata_dict.get("file_size_bytes", 0),
                    metadata_dict.get("file_type", ""),
                    metadata_dict.get("total_pages", 0),
                    metadata_dict.get("chunk_count", 0),
                    1 if metadata_dict.get("has_tables") else 0,
                    1 if metadata_dict.get("has_images") else 0,
                    metadata_dict.get("processing_time_seconds", 0),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        logger.info(f"Document registered: {filename}")

    def list_documents(self) -> List[Dict]:
        """List all active documents."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT filename, file_hash, file_size_bytes, file_type,
                       total_pages, chunk_count, has_tables, has_images,
                       processing_time_seconds, ingested_at, status
                FROM documents
                WHERE status = 'active'
                ORDER BY ingested_at DESC
                """
            ).fetchall()

        return [
            {
                "source": row["filename"],
                "file_hash": row["file_hash"],
                "file_size_bytes": row["file_size_bytes"],
                "type": row["file_type"],
                "total_pages": row["total_pages"],
                "chunks": row["chunk_count"],
                "has_tables": bool(row["has_tables"]),
                "has_images": bool(row["has_images"]),
                "processing_time_seconds": row["processing_time_seconds"],
                "ingested_at": row["ingested_at"],
                "status": row["status"],
            }
            for row in rows
        ]

    def delete(self, filename: str):
        """Soft-delete a document by setting status to 'deleted'."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'deleted' WHERE filename = ?",
                (filename,),
            )
            conn.commit()
        logger.info(f"Document soft-deleted: {filename}")

    def get_stats(self) -> Dict:
        """Get document statistics by file type."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT file_type, COUNT(*) as cnt
                FROM documents
                WHERE status = 'active'
                GROUP BY file_type
                """
            ).fetchall()

            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE status = 'active'"
            ).fetchone()["cnt"]

        return {
            "total_documents": total,
            "by_type": {row["file_type"]: row["cnt"] for row in rows},
        }
