"""
Retriever Tests
===============
Tests for the hybrid retriever and BM25.
"""

import pytest


class TestBM25:
    """Test BM25 keyword search."""

    def test_fit_and_score(self):
        """Test BM25 indexing and scoring."""
        from src.rag.retriever import BM25

        bm25 = BM25()
        documents = [
            "The company reported revenue of $5 billion",
            "Net income increased by 20% year over year",
            "The weather forecast predicts rain tomorrow",
        ]

        bm25.fit(documents)
        scores = bm25.score("revenue growth")

        assert len(scores) == 3
        assert scores[0] > scores[2]  # Revenue doc should score higher

    def test_empty_documents(self):
        """Test BM25 with no documents."""
        from src.rag.retriever import BM25

        bm25 = BM25()
        bm25.fit([])
        scores = bm25.score("test query")
        assert len(scores) == 0

    def test_tokenization(self):
        """Test BM25 tokenization."""
        from src.rag.retriever import BM25

        bm25 = BM25()
        tokens = bm25._tokenize("Revenue was $5.2B in Q3 2024!")
        assert "revenue" in tokens
        assert "q3" in tokens
        assert "2024" in tokens


class TestVectorStore:
    """Test vector store operations."""

    def test_create_and_search(self, tmp_path):
        """Test adding documents and searching."""
        import numpy as np
        from src.rag.vector_store import VectorStore

        store = VectorStore(
            dimension=4,
            index_path=str(tmp_path),
            collection_name="test",
        )

        # Add documents
        embeddings = np.random.rand(3, 4).astype(np.float32)
        contents = ["doc1", "doc2", "doc3"]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
        chunk_ids = ["c1", "c2", "c3"]

        store.add_documents(embeddings, contents, metadatas, chunk_ids)

        # Search
        query = np.random.rand(1, 4).astype(np.float32)
        results = store.search(query, k=2)

        assert len(results) == 2
        assert "content" in results[0]
        assert "score" in results[0]

    def test_save_and_load(self, tmp_path):
        """Test index persistence."""
        import numpy as np
        from src.rag.vector_store import VectorStore

        store = VectorStore(dimension=4, index_path=str(tmp_path), collection_name="persist_test")
        embeddings = np.random.rand(2, 4).astype(np.float32)
        store.add_documents(embeddings, ["a", "b"], [{"x": 1}, {"x": 2}], ["id1", "id2"])
        store.save_index()

        # Reload
        store2 = VectorStore(dimension=4, index_path=str(tmp_path), collection_name="persist_test")
        assert len(store2.content_store) == 2

    def test_get_stats(self, tmp_path):
        """Test statistics retrieval."""
        from src.rag.vector_store import VectorStore

        store = VectorStore(dimension=4, index_path=str(tmp_path), collection_name="stats_test")
        stats = store.get_stats()

        assert "total_chunks" in stats
        assert "dimension" in stats
        assert stats["dimension"] == 4
