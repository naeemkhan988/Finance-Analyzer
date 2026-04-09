"""
Vector Store Manager
====================
FAISS-based vector storage with persistence and hybrid search support.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt FAISS import
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using fallback numpy-based search.")


class VectorStore:
    """
    Production vector store with FAISS indexing.

    Features:
    - FAISS IndexFlatIP for cosine similarity (with normalized vectors)
    - Persistent storage (save/load index to disk)
    - Metadata storage alongside vectors
    - Hybrid search: vector + keyword filtering
    - Collection management (add, delete, update)
    """

    def __init__(
        self,
        dimension: int = 384,
        index_path: str = "./data/embeddings",
        collection_name: str = "financial_docs",
    ):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.collection_name = collection_name
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Storage
        self.index = None
        self.metadata_store: List[Dict] = []
        self.content_store: List[str] = []
        self.chunk_ids: List[str] = []

        # Initialize or load index
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load existing FAISS index."""
        index_file = self.index_path / f"{self.collection_name}.index"
        meta_file = self.index_path / f"{self.collection_name}_meta.json"

        if index_file.exists() and meta_file.exists():
            self._load_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index."""
        if FAISS_AVAILABLE:
            # Using Inner Product for cosine similarity (vectors should be normalized)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(
                f"Created new FAISS index (dim={self.dimension}, "
                f"collection='{self.collection_name}')"
            )
        else:
            self.index = None
            self._fallback_vectors = np.array([]).reshape(0, self.dimension)
            logger.info("Created fallback numpy vector store")

    def add_documents(
        self,
        embeddings: np.ndarray,
        contents: List[str],
        metadatas: List[Dict],
        chunk_ids: List[str],
    ):
        """
        Add document embeddings and metadata to the store.

        Args:
            embeddings: numpy array of shape (n, dimension)
            contents: List of document text contents
            metadatas: List of metadata dicts
            chunk_ids: List of unique chunk identifiers
        """
        if len(embeddings) == 0:
            return

        # Deduplicate by chunk_id
        new_mask = [cid not in self.chunk_ids for cid in chunk_ids]
        if not any(new_mask):
            logger.info("All chunks already exist in store. Skipping.")
            return

        new_embeddings = embeddings[new_mask]
        new_contents = [c for c, m in zip(contents, new_mask) if m]
        new_metadatas = [m for m, mask in zip(metadatas, new_mask) if mask]
        new_ids = [cid for cid, m in zip(chunk_ids, new_mask) if m]

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(new_embeddings.astype(np.float32))
        else:
            if len(self._fallback_vectors) == 0:
                self._fallback_vectors = new_embeddings.astype(np.float32)
            else:
                self._fallback_vectors = np.vstack(
                    [self._fallback_vectors, new_embeddings.astype(np.float32)]
                )

        self.content_store.extend(new_contents)
        self.metadata_store.extend(new_metadatas)
        self.chunk_ids.extend(new_ids)

        logger.info(
            f"Added {len(new_contents)} chunks to vector store "
            f"(total: {len(self.content_store)})"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector of shape (1, dimension)
            k: Number of results to return
            filter_metadata: Optional metadata filter dict
            score_threshold: Minimum similarity score

        Returns:
            List of result dicts with content, metadata, and score
        """
        if len(self.content_store) == 0:
            return []

        # Ensure proper shape
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        if FAISS_AVAILABLE and self.index is not None:
            search_k = min(k * 3, self.index.ntotal)
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), search_k
            )
            scores = scores[0]
            indices = indices[0]
        else:
            similarities = np.dot(
                self._fallback_vectors, query_embedding.T
            ).flatten()
            search_k = min(k * 3, len(similarities))
            indices = np.argsort(similarities)[::-1][:search_k]
            scores = similarities[indices]

        # Build results with filtering
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.content_store):
                continue
            if score < score_threshold:
                continue

            metadata = self.metadata_store[idx]

            # Apply metadata filter
            if filter_metadata:
                if not self._matches_filter(metadata, filter_metadata):
                    continue

            results.append(
                {
                    "content": self.content_store[idx],
                    "metadata": metadata,
                    "score": float(score),
                    "chunk_id": self.chunk_ids[idx],
                }
            )

            if len(results) >= k:
                break

        return results

    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def save_index(self):
        """Persist index and metadata to disk."""
        try:
            if FAISS_AVAILABLE and self.index is not None:
                index_file = self.index_path / f"{self.collection_name}.index"
                faiss.write_index(self.index, str(index_file))
            else:
                np_file = self.index_path / f"{self.collection_name}_vectors.npy"
                np.save(str(np_file), self._fallback_vectors)

            meta_file = self.index_path / f"{self.collection_name}_meta.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "contents": self.content_store,
                        "metadatas": self.metadata_store,
                        "chunk_ids": self.chunk_ids,
                        "dimension": self.dimension,
                    },
                    f,
                    ensure_ascii=False,
                    default=str,
                )

            logger.info(
                f"Saved vector store: {len(self.content_store)} chunks "
                f"to '{self.index_path}'"
            )
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def _load_index(self):
        """Load index and metadata from disk."""
        try:
            meta_file = self.index_path / f"{self.collection_name}_meta.json"
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.content_store = data["contents"]
            self.metadata_store = data["metadatas"]
            self.chunk_ids = data["chunk_ids"]
            self.dimension = data.get("dimension", self.dimension)

            index_file = self.index_path / f"{self.collection_name}.index"
            if FAISS_AVAILABLE and index_file.exists():
                self.index = faiss.read_index(str(index_file))
                logger.info(
                    f"Loaded FAISS index: {self.index.ntotal} vectors "
                    f"from '{index_file}'"
                )
            else:
                np_file = self.index_path / f"{self.collection_name}_vectors.npy"
                if np_file.exists():
                    self._fallback_vectors = np.load(str(np_file))
                else:
                    self._fallback_vectors = np.array([]).reshape(
                        0, self.dimension
                    )

            logger.info(
                f"Loaded vector store: {len(self.content_store)} chunks"
            )
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self._create_new_index()

    def delete_by_source(self, source_name: str):
        """
        Delete all chunks from a specific source document.
        Requires rebuilding the index.
        """
        keep_mask = [
            m.get("source") != source_name for m in self.metadata_store
        ]

        if all(keep_mask):
            logger.info(f"No chunks found for source '{source_name}'")
            return

        self.content_store = [
            c for c, k in zip(self.content_store, keep_mask) if k
        ]
        self.metadata_store = [
            m for m, k in zip(self.metadata_store, keep_mask) if k
        ]
        self.chunk_ids = [
            cid for cid, k in zip(self.chunk_ids, keep_mask) if k
        ]

        logger.info(
            f"Deleted chunks from '{source_name}'. "
            f"Remaining: {len(self.content_store)} chunks. "
            f"Note: Index rebuild required with re-embedding."
        )

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        sources = set()
        for m in self.metadata_store:
            sources.add(m.get("source", "unknown"))

        total_vectors = 0
        if FAISS_AVAILABLE and self.index is not None:
            total_vectors = self.index.ntotal
        elif hasattr(self, "_fallback_vectors"):
            total_vectors = len(self._fallback_vectors)

        return {
            "total_chunks": len(self.content_store),
            "total_vectors": total_vectors,
            "dimension": self.dimension,
            "unique_sources": list(sources),
            "source_count": len(sources),
            "collection": self.collection_name,
            "faiss_available": FAISS_AVAILABLE,
        }

    def clear(self):
        """Clear all data from the store."""
        self._create_new_index()
        self.metadata_store = []
        self.content_store = []
        self.chunk_ids = []
        logger.info("Vector store cleared")
