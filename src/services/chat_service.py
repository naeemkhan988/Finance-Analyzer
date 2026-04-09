"""
Chat Service
=============
Manages conversational interactions with session support.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.utils.helpers import generate_session_id

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service layer for chat/conversation management.

    Features:
    - Session management
    - Conversation history
    - Context-aware responses
    - Multi-session support
    """

    def __init__(self, max_history_per_session: int = 50):
        self.max_history = max_history_per_session
        self._sessions: Dict[str, Dict] = {}

    def create_session(self, user_id: str = "anonymous") -> Dict:
        """Create a new chat session."""
        session_id = generate_session_id()

        self._sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
            "active": True,
        }

        logger.info(f"Chat session created: {session_id}")
        return {"session_id": session_id}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ):
        """Add a message to a session's history."""
        if session_id not in self._sessions:
            self.create_session()

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        self._sessions[session_id]["messages"].append(message)

        # Trim history
        if len(self._sessions[session_id]["messages"]) > self.max_history:
            self._sessions[session_id]["messages"] = (
                self._sessions[session_id]["messages"][-self.max_history:]
            )

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a session."""
        if session_id not in self._sessions:
            return []

        messages = self._sessions[session_id]["messages"]
        return messages[-limit:]

    def get_context_summary(self, session_id: str, max_messages: int = 3) -> str:
        """Get a text summary of recent context for RAG augmentation."""
        history = self.get_history(session_id, limit=max_messages)

        if not history:
            return ""

        parts = []
        for msg in history:
            role = msg["role"].upper()
            content = msg["content"][:200]
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def clear_session(self, session_id: str) -> Dict:
        """Clear a session's history."""
        if session_id in self._sessions:
            self._sessions[session_id]["messages"] = []
            return {"success": True, "message": "Session cleared"}
        return {"success": False, "error": "Session not found"}

    def delete_session(self, session_id: str) -> Dict:
        """Delete a session entirely."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return {"success": True, "message": "Session deleted"}
        return {"success": False, "error": "Session not found"}

    def list_sessions(self) -> List[Dict]:
        """List all active sessions."""
        sessions = []
        for sid, data in self._sessions.items():
            sessions.append({
                "session_id": sid,
                "user_id": data["user_id"],
                "created_at": data["created_at"],
                "message_count": len(data["messages"]),
                "active": data["active"],
            })
        return sessions
