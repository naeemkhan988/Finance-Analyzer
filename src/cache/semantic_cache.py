# Semantic cache — cosine similarity-based query cache with SQLite persistence.
"""
Semantic Cache
==============
High-performance semantic query cache using cosine similarity.
Avoids redundant LLM calls for similar questions.
"""

import json
import sqlite3
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic query cache with cosine similarity matching.

    Features:
    - Cosine similarity-based cache lookup
    - Configurable similarity threshold
    - TTL-based expiration
    - Hit count tracking
    - Automatic pruning of expired/low-value entries
    """

    def __init__(
        self,
        db_path: str = "./data/semantic_cache.db",
        similarity_threshold: float = 0.92,
        ttl_hours: int = 24,
        max_entries: int = 1000,
    ):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self.max_entries = max_entries

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory embedding cache for fast similarity search
        self._cache_embeddings: List[np.ndarray] = []
        self._cache_ids: List[int] = []
        self._load_embeddings()

    def _init_db(self):
        """Create semantic_cache table."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    query_embedding BLOB NOT NULL,
                    response TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    hit_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            conn.commit()
        logger.info(f"SemanticCache initialized: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_embeddings(self):
        """Load all non-expired embeddings into memory for fast similarity search."""
        self._cache_embeddings = []
        self._cache_ids = []
        now = datetime.utcnow().isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, query_embedding FROM semantic_cache
                WHERE expires_at > ?
                """,
                (now,),
            ).fetchall()

        for row in rows:
            embedding = np.frombuffer(row["query_embedding"], dtype=np.float32)
            self._cache_embeddings.append(embedding)
            self._cache_ids.append(row["id"])

        logger.info(f"Loaded {len(self._cache_embeddings)} cached embeddings into memory")

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[Dict]:
        """
        Check cache for a semantically similar query.

        Args:
            query: The query text
            query_embedding: The query embedding vector (1D or 2D)

        Returns:
            Cached response dict if similarity exceeds threshold, None otherwise.
        """
        if not self._cache_embeddings:
            return None

        # Ensure 1D
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()

        # Compute cosine similarity against all cached embeddings
        cache_matrix = np.array(self._cache_embeddings)
        similarities = np.dot(cache_matrix, query_embedding)

        # Normalize (embeddings should already be normalized, but be safe)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            cache_norms = np.linalg.norm(cache_matrix, axis=1)
            mask = cache_norms > 0
            similarities[mask] = similarities[mask] / (cache_norms[mask] * query_norm)

        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])

        if best_similarity >= self.similarity_threshold:
            cache_id = self._cache_ids[best_idx]

            with self._connect() as conn:
                # Increment hit count
                conn.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = ?",
                    (cache_id,),
                )
                conn.commit()

                row = conn.execute(
                    "SELECT response, sources, metadata FROM semantic_cache WHERE id = ?",
                    (cache_id,),
                ).fetchone()

            if row:
                logger.info(
                    f"Cache HIT (similarity={best_similarity:.3f}) for: '{query[:50]}...'"
                )
                return {
                    "answer": row["response"],
                    "sources": json.loads(row["sources"]),
                    "metadata": {
                        **json.loads(row["metadata"]),
                        "cache_hit": True,
                        "cache_similarity": round(best_similarity, 4),
                    },
                }

        return None

    def set(
        self,
        query: str,
        query_embedding: np.ndarray,
        response: Dict,
    ):
        """
        Store a new query-response pair in the cache.

        Args:
            query: The query text
            query_embedding: The query embedding vector
            response: The full response dict (must contain 'answer', 'sources', 'metadata')
        """
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()

        now = datetime.utcnow()
        expires = now + timedelta(hours=self.ttl_hours)

        embedding_blob = query_embedding.astype(np.float32).tobytes()

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO semantic_cache
                    (query_text, query_embedding, response, sources, metadata,
                     hit_count, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    query,
                    embedding_blob,
                    response.get("answer", ""),
                    json.dumps(response.get("sources", []), default=str),
                    json.dumps(response.get("metadata", {}), default=str),
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
            conn.commit()
            new_id = cursor.lastrowid

        # Update in-memory cache
        self._cache_embeddings.append(query_embedding.astype(np.float32))
        self._cache_ids.append(new_id)

        # Prune if over max entries
        if len(self._cache_ids) > self.max_entries:
            self._prune_cache()

        logger.debug(f"Cached response for: '{query[:50]}...'")

    def _prune_cache(self):
        """Remove expired entries and lowest hit count entries if over max."""
        now = datetime.utcnow().isoformat()

        with self._connect() as conn:
            # Delete expired entries
            conn.execute(
                "DELETE FROM semantic_cache WHERE expires_at <= ?",
                (now,),
            )

            # Count remaining
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM semantic_cache"
            ).fetchone()["cnt"]

            # Delete lowest hit count entries if still over max
            if count > self.max_entries:
                excess = count - self.max_entries
                conn.execute(
                    """
                    DELETE FROM semantic_cache WHERE id IN (
                        SELECT id FROM semantic_cache
                        ORDER BY hit_count ASC, created_at ASC
                        LIMIT ?
                    )
                    """,
                    (excess,),
                )

            conn.commit()

        # Reload in-memory cache
        self._load_embeddings()
        logger.info(f"Cache pruned. Active entries: {len(self._cache_ids)}")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM semantic_cache"
            ).fetchone()["cnt"]

            hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) as total_hits FROM semantic_cache"
            ).fetchone()["total_hits"]

            avg_hits = conn.execute(
                "SELECT COALESCE(AVG(hit_count), 0) as avg_hits FROM semantic_cache"
            ).fetchone()["avg_hits"]

        return {
            "cached_queries": total,
            "total_cache_hits": hits,
            "avg_hits_per_entry": round(float(avg_hits), 2),
        }
