"""
API Client
==========
HTTP client for communicating with the FastAPI backend.
"""

import os
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class APIClient:
    """Client for the Finance Analyzer REST API."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv(
            "API_BASE_URL", "http://localhost:8000/api"
        )
        self.timeout = 60

    def health_check(self) -> Dict:
        """Check API health."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def upload_document(self, file) -> Dict:
        """Upload a document to the API."""
        try:
            files = {"file": (file.filename, file.read(), file.content_type)}
            resp = requests.post(
                f"{self.base_url}/upload",
                files=files,
                timeout=120,
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"success": False, "message": str(e)}

    def query(
        self,
        question: str,
        session_id: str = "default",
        k: int = 5,
        filter_source: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> Dict:
        """Send a query to the API."""
        try:
            resp = requests.post(
                f"{self.base_url}/query",
                json={
                    "question": question,
                    "session_id": session_id,
                    "k": k,
                    "filter_source": filter_source,
                    "use_hybrid": use_hybrid,
                },
                timeout=self.timeout,
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"success": False, "answer": f"API error: {str(e)}"}

    def list_documents(self) -> List[Dict]:
        """List uploaded documents."""
        try:
            resp = requests.get(f"{self.base_url}/documents", timeout=10)
            data = resp.json()
            return data.get("documents", [])
        except Exception:
            return []

    def delete_document(self, filename: str) -> Dict:
        """Delete a document."""
        try:
            resp = requests.delete(
                f"{self.base_url}/documents/{filename}", timeout=10
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_history(self, session_id: str = "default") -> List[Dict]:
        """Get conversation history."""
        try:
            resp = requests.get(
                f"{self.base_url}/history/{session_id}", timeout=10
            )
            return resp.json().get("history", [])
        except Exception:
            return []

    def clear_history(self, session_id: str = "default") -> Dict:
        """Clear conversation history."""
        try:
            resp = requests.delete(
                f"{self.base_url}/history/{session_id}", timeout=10
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict:
        """Get system statistics."""
        try:
            resp = requests.get(f"{self.base_url}/stats", timeout=10)
            return resp.json()
        except Exception:
            return {}

    def get_providers(self) -> Dict:
        """Get LLM provider information."""
        try:
            resp = requests.get(f"{self.base_url}/providers", timeout=5)
            return resp.json()
        except Exception:
            return {}
