# Conversation store — SQLite-backed persistent conversation history storage.
"""
Conversation Store
==================
SQLite-backed persistent storage for conversation history.
Supports session management, analytics, and history retrieval.
"""

import json
import sqlite3
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationStore:
    """
    Persistent conversation storage using SQLite.

    Features:
    - Session-based conversation tracking
    - Exchange history with sources and metadata
    - Session management (list, delete)
    - Analytics queries (totals, frequencies)
    - Indexed on session_id for fast lookups
    """

    def __init__(self, db_path: str = "./data/conversations.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create conversations table if not exists."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session_id
                ON conversations (session_id)
            """)
            conn.commit()
        logger.info(f"ConversationStore initialized: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ):
        """Save a question-answer exchange to the database."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations
                    (session_id, question, answer, sources, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    answer,
                    json.dumps(sources or [], default=str),
                    json.dumps(metadata or {}, default=str),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, answer, sources, metadata, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        history = []
        for row in reversed(rows):
            history.append({
                "question": row["question"],
                "answer": row["answer"],
                "sources": json.loads(row["sources"]),
                "metadata": json.loads(row["metadata"]),
                "timestamp": row["timestamp"],
            })
        return history

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions with exchange counts."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id,
                       COUNT(*) as exchange_count,
                       MIN(timestamp) as first_exchange,
                       MAX(timestamp) as last_exchange
                FROM conversations
                GROUP BY session_id
                ORDER BY last_exchange DESC
                """
            ).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "exchange_count": row["exchange_count"],
                "first_exchange": row["first_exchange"],
                "last_exchange": row["last_exchange"],
            }
            for row in rows
        ]

    def delete_session(self, session_id: str):
        """Delete all exchanges for a session."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
        logger.info(f"Deleted session: {session_id}")

    def get_analytics(self) -> Dict:
        """Get conversation analytics."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM conversations"
            ).fetchone()["cnt"]

            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) as cnt FROM conversations"
            ).fetchone()["cnt"]

            recent = conn.execute(
                """
                SELECT question FROM conversations
                ORDER BY id DESC LIMIT 20
                """
            ).fetchall()

        return {
            "total_exchanges": total,
            "unique_sessions": sessions,
            "recent_questions": [r["question"] for r in recent],
        }
