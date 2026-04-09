"""
Embedding Manager
=================
Handles text embedding generation using sentence-transformers.
Supports batch processing and caching for production efficiency.
"""

import logging
import numpy as np
from typing import List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global model cache
_embedding_model = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load and cache the embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded successfully. Dimension: {_embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _embedding_model


class EmbeddingManager:
    """
    Production embedding manager with batching and caching.

    Features:
    - Lazy model loading (loads once, reuses)
    - Batch embedding for efficiency
    - Numpy array output for FAISS compatibility
    - Embedding normalization for cosine similarity
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )

            logger.debug(
                f"Generated {len(embeddings)} embeddings "
                f"(dim={embeddings.shape[1] if len(embeddings.shape) > 1 else 0})"
            )
            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.

        Returns:
            numpy array of shape (1, embedding_dim)
        """
        embedding = self.embed_texts([query])
        return embedding

    def compute_similarity(
        self, query_embedding: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.

        Args:
            query_embedding: Shape (1, dim)
            doc_embeddings: Shape (n, dim)

        Returns:
            Similarity scores of shape (n,)
        """
        if self.normalize:
            # If normalized, dot product = cosine similarity
            return np.dot(doc_embeddings, query_embedding.T).flatten()

        # Manual cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(
            doc_embeddings, axis=1, keepdims=True
        )
        return np.dot(doc_norms, query_norm.T).flatten()
