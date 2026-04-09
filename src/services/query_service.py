"""
Query Service
=============
Business logic for query processing and response management.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryService:
    """
    Service layer for query operations.

    Handles:
    - Query validation and preprocessing
    - RAG pipeline invocation
    - Response formatting
    - Query history tracking
    """

    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self._query_history: List[Dict] = []

    def process_query(
        self,
        question: str,
        session_id: str = "default",
        k: int = 5,
        filter_source: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> Dict:
        """
        Process a user query through the RAG pipeline.

        Args:
            question: User's question
            session_id: Session ID for conversation memory
            k: Number of context chunks to retrieve
            filter_source: Optional document filter
            use_hybrid: Whether to use hybrid search

        Returns:
            Formatted response dict
        """
        # Validate query
        validation = self._validate_query(question)
        if not validation["valid"]:
            return {
                "success": False,
                "error": validation["error"],
            }

        # Clean and preprocess query
        cleaned_query = self._preprocess_query(question)

        # Check if RAG pipeline is initialized
        if self.rag_pipeline is None:
            return {
                "success": False,
                "error": "RAG pipeline not initialized. No documents loaded.",
            }

        # Execute query
        result = self.rag_pipeline.query(
            question=cleaned_query,
            session_id=session_id,
            k=k,
            filter_source=filter_source,
            use_hybrid=use_hybrid,
        )

        # Track query
        self._track_query(question, result)

        return {
            "success": True,
            **result,
        }

    def _validate_query(self, question: str) -> Dict:
        """Validate a query before processing."""
        if not question or not question.strip():
            return {"valid": False, "error": "Question cannot be empty"}

        if len(question) > 2000:
            return {"valid": False, "error": "Question too long (max 2000 chars)"}

        if len(question.strip()) < 3:
            return {"valid": False, "error": "Question too short"}

        return {"valid": True}

    def _preprocess_query(self, question: str) -> str:
        """Clean and preprocess the query."""
        # Strip whitespace
        cleaned = question.strip()

        # Ensure ends with question mark for questions
        if not cleaned.endswith(("?", ".", "!")):
            cleaned += "?"

        return cleaned

    def _track_query(self, question: str, result: Dict):
        """Track query for analytics."""
        self._query_history.append({
            "question": question,
            "answer_preview": result.get("answer", "")[:100],
            "sources_count": len(result.get("sources", [])),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": result.get("metadata", {}),
        })

        # Keep last 100 queries
        if len(self._query_history) > 100:
            self._query_history = self._query_history[-100:]

    def get_query_history(self, limit: int = 20) -> List[Dict]:
        """Get recent query history."""
        return self._query_history[-limit:]

    def get_query_stats(self) -> Dict:
        """Get query analytics."""
        if not self._query_history:
            return {"total_queries": 0}

        return {
            "total_queries": len(self._query_history),
            "recent_queries": len(self._query_history[-10:]),
        }
