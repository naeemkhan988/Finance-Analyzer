"""
Metadata Database
=================
SQLAlchemy-based metadata storage for documents and queries.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# SQLAlchemy models (lazy import to avoid hard dependency)
_db = None


def get_db():
    """Get or create the SQLAlchemy database instance."""
    global _db
    if _db is None:
        try:
            from flask_sqlalchemy import SQLAlchemy
            _db = SQLAlchemy()
        except ImportError:
            logger.warning("Flask-SQLAlchemy not available. Using in-memory storage.")
    return _db


class MetadataStore:
    """
    In-memory metadata storage (fallback when SQLAlchemy is not available).

    Stores document metadata, query logs, and user sessions.
    """

    def __init__(self):
        self._documents: Dict[str, Dict] = {}
        self._queries: List[Dict] = []

    def add_document(self, doc_id: str, metadata: Dict):
        """Store document metadata."""
        self._documents[doc_id] = {
            **metadata,
            "doc_id": doc_id,
            "created_at": datetime.utcnow().isoformat(),
        }

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document metadata."""
        return self._documents.get(doc_id)

    def list_documents(self) -> List[Dict]:
        """List all document metadata."""
        return list(self._documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete document metadata."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def log_query(self, query: str, result: Dict, session_id: str = ""):
        """Log a query for analytics."""
        self._queries.append({
            "query": query,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "provider": result.get("metadata", {}).get("provider", "unknown"),
            "latency": result.get("metadata", {}).get("query_time_seconds", 0),
            "sources_count": len(result.get("sources", [])),
        })

        # Keep last 1000 queries
        if len(self._queries) > 1000:
            self._queries = self._queries[-1000:]

    def get_query_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent query logs."""
        return self._queries[-limit:]

    def get_stats(self) -> Dict:
        """Get metadata store statistics."""
        return {
            "total_documents": len(self._documents),
            "total_queries_logged": len(self._queries),
        }
