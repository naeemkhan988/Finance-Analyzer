"""
Embedding Tests
===============
Tests for the embedding manager.
"""

import pytest
import numpy as np


class TestEmbeddingManager:
    """Test embedding generation."""

    @pytest.fixture
    def manager(self):
        """Create embedding manager (uses lightweight model)."""
        from src.rag.embeddings import EmbeddingManager
        return EmbeddingManager(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
        )

    @pytest.mark.slow
    def test_embed_texts(self, manager):
        """Test batch embedding generation."""
        texts = ["Revenue increased by 10%", "Net income was $5 million"]
        embeddings = manager.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # MiniLM dimension

    @pytest.mark.slow
    def test_embed_query(self, manager):
        """Test single query embedding."""
        embedding = manager.embed_query("What was the revenue?")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)

    @pytest.mark.slow
    def test_compute_similarity(self, manager):
        """Test cosine similarity computation."""
        texts = ["Revenue growth", "Income statement", "Weather forecast"]
        embeddings = manager.embed_texts(texts)
        query_emb = manager.embed_query("Revenue growth trends")

        similarities = manager.compute_similarity(query_emb, embeddings)
        assert len(similarities) == 3
        # Revenue growth should be most similar
        assert similarities[0] > similarities[2]

    def test_empty_texts(self, manager):
        """Test embedding empty list."""
        embeddings = manager.embed_texts([])
        assert len(embeddings) == 0
