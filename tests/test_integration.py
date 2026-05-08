# Integration tests — full pipeline, semantic cache, and rate limiter tests.
"""
Integration Tests
=================
End-to-end tests for the RAG pipeline, semantic cache, and rate limiter.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestFullPipeline:
    """Test the complete RAG pipeline end-to-end."""

    @pytest.fixture(scope="class")
    def pipeline(self, tmp_path_factory):
        """Create a pipeline instance with temporary storage."""
        tmp_dir = tmp_path_factory.mktemp("pipeline_test")
        os.environ["CACHE_SIMILARITY_THRESHOLD"] = "0.92"
        os.environ["CACHE_TTL_HOURS"] = "24"

        from src.rag.pipeline import RAGPipeline

        pipe = RAGPipeline(
            chunk_size=256,
            chunk_overlap=64,
            index_path=str(tmp_dir / "embeddings"),
            collection_name="test_docs",
        )
        # Override DB paths to use temp dir
        from src.database.conversation_store import ConversationStore
        from src.database.document_registry import DocumentRegistry
        from src.cache.semantic_cache import SemanticCache

        pipe.conversation_store = ConversationStore(
            db_path=str(tmp_dir / "conv.db")
        )
        pipe.document_registry = DocumentRegistry(
            db_path=str(tmp_dir / "docs.db")
        )
        pipe.semantic_cache = SemanticCache(
            db_path=str(tmp_dir / "cache.db"),
            similarity_threshold=0.92,
            ttl_hours=24,
        )
        return pipe

    @pytest.fixture()
    def sample_file(self, tmp_path):
        """Create a sample text file for ingestion."""
        content = (
            "Annual Financial Report 2024\n\n"
            "Revenue: $10.5 billion for the fiscal year.\n"
            "Net income increased by 15% year-over-year to $2.3 billion.\n"
            "Operating expenses totaled $7.1 billion.\n"
            "Gross margin was 32.5%, up from 30.1% in the prior year.\n"
            "The company declared dividends of $1.50 per share.\n"
            "Total assets: $45.2 billion. Total liabilities: $18.7 billion.\n"
            "Free cash flow reached $3.8 billion.\n"
        )
        file_path = tmp_path / "test_report.txt"
        file_path.write_text(content)
        return str(file_path)

    def test_ingest_creates_chunks(self, pipeline, sample_file):
        """Test that document ingestion creates chunks."""
        result = pipeline.ingest_document(sample_file)
        assert result["success"] is True
        assert result.get("chunks_created", 0) > 0

    def test_query_returns_relevant_answer(self, pipeline, sample_file):
        """Test that querying returns relevant answers."""
        # Ensure document is ingested
        if pipeline.vector_store.get_stats()["total_chunks"] == 0:
            pipeline.ingest_document(sample_file)

        result = pipeline.query(
            question="What was the revenue?",
            session_id="test_session",
        )
        answer = result.get("answer", "").lower()
        assert "10.5" in answer or "billion" in answer or "revenue" in answer

    def test_conversation_persists(self, pipeline, sample_file):
        """Test that conversation history is persisted."""
        session_id = "persist_test"

        # Ensure document is ingested
        if pipeline.vector_store.get_stats()["total_chunks"] == 0:
            pipeline.ingest_document(sample_file)

        pipeline.query(question="What was revenue?", session_id=session_id)
        pipeline.query(question="What about net income?", session_id=session_id)

        history = pipeline.get_conversation_history(session_id)
        assert len(history) >= 2

    def test_delete_removes_chunks(self, pipeline, sample_file):
        """Test that deleting a document removes its chunks."""
        # Ingest
        result = pipeline.ingest_document(sample_file)
        assert result["success"]
        source = result["source"]
        chunks_before = pipeline.vector_store.get_stats()["total_chunks"]

        # Delete
        pipeline.delete_document(source)
        chunks_after = pipeline.vector_store.get_stats()["total_chunks"]
        assert chunks_after < chunks_before

    def test_query_after_delete_returns_empty_sources(self, pipeline, sample_file):
        """Test that queries return empty sources after all documents deleted."""
        # Ingest then delete
        result = pipeline.ingest_document(sample_file)
        source = result["source"]
        pipeline.delete_document(source)

        # Query with no documents
        result = pipeline.query(
            question="What was the revenue?",
            session_id="empty_test",
        )
        sources = result.get("sources", [])
        assert len(sources) == 0


class TestSemanticCache:
    """Test the semantic cache."""

    @pytest.fixture()
    def cache(self, tmp_path):
        from src.cache.semantic_cache import SemanticCache
        return SemanticCache(
            db_path=str(tmp_path / "test_cache.db"),
            similarity_threshold=0.90,
            ttl_hours=24,
        )

    def test_cache_hit_on_identical_embedding(self, cache):
        """Test cache hit when embedding is identical."""
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        response = {
            "answer": "Revenue was $10.5 billion",
            "sources": [{"document": "test.pdf", "page": 1}],
            "metadata": {"provider": "groq"},
        }
        cache.set("What was revenue?", embedding, response)

        result = cache.get("What was revenue?", embedding)
        assert result is not None
        assert result["metadata"].get("cache_hit") is True

    def test_cache_miss_on_different_embedding(self, cache):
        """Test cache miss when embedding is orthogonal."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0  # unit vector along dim 0

        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0  # unit vector along dim 1 (orthogonal)

        response = {"answer": "test", "sources": [], "metadata": {}}
        cache.set("query1", emb1, response)

        result = cache.get("query2", emb2)
        assert result is None

    def test_hit_count_increments(self, cache):
        """Test that hit count increases on repeated cache hits."""
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        response = {"answer": "test", "sources": [], "metadata": {}}
        cache.set("test query", embedding, response)

        # Hit 3 times
        for _ in range(3):
            cache.get("test query", embedding)

        stats = cache.get_stats()
        assert stats["total_cache_hits"] >= 3

    def test_expired_entries_not_returned(self, tmp_path):
        """Test that expired entries are not returned."""
        from src.cache.semantic_cache import SemanticCache

        cache = SemanticCache(
            db_path=str(tmp_path / "expire_cache.db"),
            similarity_threshold=0.90,
            ttl_hours=0,  # Expire immediately
        )

        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        response = {"answer": "expired", "sources": [], "metadata": {}}
        cache.set("test", embedding, response)

        # Force reload to respect expiry
        cache._load_embeddings()

        result = cache.get("test", embedding)
        assert result is None


class TestRateLimiter:
    """Test the rate limiter."""

    @pytest.fixture()
    def limiter(self):
        from src.auth.rate_limiter import RateLimiter
        return RateLimiter()

    def test_allows_requests_under_limit(self, limiter):
        """Test that requests under the limit are allowed."""
        allowed, info = limiter.check("user1", "free")
        assert allowed is True
        assert "remaining" in info

    def test_blocks_requests_over_minute_limit(self, limiter):
        """Test that requests over the per-minute limit are blocked."""
        identifier = "rate_test_user"

        # Free tier: 5/min
        for i in range(5):
            allowed, _ = limiter.check(identifier, "free")
            assert allowed is True

        # 6th request should be blocked
        allowed, info = limiter.check(identifier, "free")
        assert allowed is False
        assert "retry_after" in info

    def test_different_identifiers_independent(self, limiter):
        """Test that different identifiers have independent rate limits."""
        # Exhaust user_a's limit
        for _ in range(5):
            limiter.check("user_a", "free")

        # user_b should still be allowed
        allowed, _ = limiter.check("user_b", "free")
        assert allowed is True
