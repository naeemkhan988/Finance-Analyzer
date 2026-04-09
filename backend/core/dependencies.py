"""
Dependencies
=============
FastAPI dependency injection for shared resources.
"""

import logging
from functools import lru_cache

from backend.core.config import settings

logger = logging.getLogger(__name__)

_rag_pipeline = None


def get_rag_pipeline():
    """Get or create the RAG pipeline singleton."""
    global _rag_pipeline

    if _rag_pipeline is None:
        try:
            from src.rag.pipeline import RAGPipeline

            _rag_pipeline = RAGPipeline(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                embedding_model=settings.EMBEDDING_MODEL,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
                index_path=settings.FAISS_INDEX_PATH,
                collection_name=settings.FAISS_INDEX_NAME,
            )
            logger.info("RAG Pipeline created successfully")
        except Exception as e:
            logger.error(f"Failed to create RAG Pipeline: {e}")
            raise

    return _rag_pipeline
