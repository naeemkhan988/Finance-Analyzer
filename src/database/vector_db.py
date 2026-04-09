"""
Vector Database Manager
=======================
Abstraction layer for vector database operations.
Supports FAISS with option to extend to other backends.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VectorDB:
    """
    High-level vector database manager.

    Provides a clean interface over the VectorStore with
    additional features like collection management and stats.
    """

    def __init__(
        self,
        index_path: str = "./data/embeddings",
        dimension: int = 384,
        default_collection: str = "financial_docs",
    ):
        self.index_path = index_path
        self.dimension = dimension
        self.default_collection = default_collection
        self._collections: Dict[str, VectorStore] = {}

        # Initialize default collection
        self.get_collection(default_collection)

    def get_collection(self, name: str = None) -> VectorStore:
        """Get or create a vector store collection."""
        name = name or self.default_collection

        if name not in self._collections:
            self._collections[name] = VectorStore(
                dimension=self.dimension,
                index_path=self.index_path,
                collection_name=name,
            )
            logger.info(f"Collection '{name}' initialized")

        return self._collections[name]

    def list_collections(self) -> List[Dict]:
        """List all collections with stats."""
        collections = []
        for name, store in self._collections.items():
            stats = store.get_stats()
            collections.append({
                "name": name,
                "chunks": stats["total_chunks"],
                "sources": stats["source_count"],
            })
        return collections

    def delete_collection(self, name: str) -> Dict:
        """Delete a collection and its data."""
        if name in self._collections:
            self._collections[name].clear()
            del self._collections[name]

            # Remove files
            index_dir = Path(self.index_path)
            for f in index_dir.glob(f"{name}*"):
                f.unlink()

            return {"success": True, "message": f"Collection '{name}' deleted"}
        return {"success": False, "error": f"Collection '{name}' not found"}

    def get_total_stats(self) -> Dict:
        """Get aggregate stats across all collections."""
        total_chunks = 0
        total_sources = 0

        for store in self._collections.values():
            stats = store.get_stats()
            total_chunks += stats["total_chunks"]
            total_sources += stats["source_count"]

        return {
            "total_collections": len(self._collections),
            "total_chunks": total_chunks,
            "total_sources": total_sources,
        }
