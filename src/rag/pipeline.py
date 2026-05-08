# RAG pipeline — full end-to-end orchestrator with persistent storage, multimodal, and caching.
"""
RAG Pipeline
=============
Full end-to-end RAG pipeline orchestrator.
Coordinates document processing, retrieval, generation,
persistent storage, multimodal fusion, and semantic caching.
"""

import os
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime

from src.rag.document_loader import DocumentProcessor, DocumentChunk
from src.rag.embeddings import EmbeddingManager
from src.rag.vector_store import VectorStore
from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker
from src.rag.generator import ResponseGenerator
from src.models.llm import LLMRouter
from src.database.conversation_store import ConversationStore
from src.database.document_registry import DocumentRegistry
from src.cache.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production RAG pipeline orchestrating the full workflow.

    Pipeline:
    1. Document Upload → Parse → Multimodal Fusion → Chunk → Embed → Store
    2. Query → Cache Check → Expand → Hybrid Retrieve → Re-rank → Generate → Cache → Cite

    Features:
    - End-to-end document processing pipeline
    - Multimodal fusion (tables, images, OCR)
    - Hybrid retrieval (vector + keyword)
    - Multi-provider LLM generation with failover
    - Source citation and provenance tracking
    - Persistent conversation memory (SQLite)
    - Persistent document registry (SQLite)
    - Semantic query caching
    - Performance metrics and analytics
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        index_path: str = "./data/embeddings",
        collection_name: str = "financial_docs",
    ):
        logger.info("Initializing RAG Pipeline...")

        # Core components
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
        )

        self.vector_store = VectorStore(
            dimension=embedding_dimension,
            index_path=index_path,
            collection_name=collection_name,
        )

        self.retriever = HybridRetriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
        )

        self.reranker = Reranker()

        self.llm_router = LLMRouter()

        self.generator = ResponseGenerator(llm_router=self.llm_router)

        # Persistent storage
        db_dir = "./data"
        self.conversation_store = ConversationStore(
            db_path=os.path.join(db_dir, "conversations.db")
        )
        self.document_registry = DocumentRegistry(
            db_path=os.path.join(db_dir, "documents.db")
        )

        # Semantic cache
        self.semantic_cache = SemanticCache(
            db_path=os.path.join(db_dir, "semantic_cache.db"),
            similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")),
            ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        )

        # Multimodal fusion (optional, fails gracefully)
        self.multimodal_fusion = None
        try:
            from src.multimodal.multimodal_fusion import MultimodalFusion
            self.multimodal_fusion = MultimodalFusion(
                enable_ocr=True,
                enable_images=True,
                enable_tables=True,
            )
            logger.info("MultimodalFusion initialized")
        except Exception as e:
            logger.warning(f"MultimodalFusion unavailable: {e}")

        # In-memory conversation cache (fast path, backed by SQLite)
        self.conversations: Dict[str, List[Dict]] = {}

        # Analytics
        self._stats = {
            "documents_processed": 0,
            "total_queries": 0,
            "avg_query_time": 0,
            "total_chunks_stored": 0,
        }

        logger.info(
            f"RAG Pipeline initialized. "
            f"Vector store: {self.vector_store.get_stats()['total_chunks']} chunks. "
            f"LLM: {self.llm_router.active_provider or 'fallback'}"
        )

    def ingest_document(self, file_path: str) -> Dict:
        """
        Ingest a document into the RAG system.

        Pipeline: Load → Multimodal Fusion → Chunk → Embed → Store → Register
        """
        start_time = time.time()
        logger.info(f"Ingesting document: {file_path}")

        try:
            # 1. Load and parse
            pages = self.doc_processor.load_document(file_path)
            if not pages:
                return {
                    "success": False,
                    "error": "Failed to extract content from document",
                }

            # 2. Multimodal fusion for PDFs
            table_count = 0
            image_count = 0
            if self.multimodal_fusion and file_path.lower().endswith(".pdf"):
                try:
                    mm_result = self.multimodal_fusion.process_document(file_path)
                    fused_chunks = mm_result.get("fused_chunks", [])
                    table_count = mm_result.get("metadata", {}).get("table_count", 0)
                    image_count = mm_result.get("metadata", {}).get("image_count", 0)

                    # Append fused chunks as additional pages
                    for fc in fused_chunks:
                        source = pages[0]["metadata"]["source"] if pages else "unknown"
                        pages.append({
                            "content": fc.get("content", ""),
                            "metadata": {
                                **fc.get("metadata", {}),
                                "source": fc.get("metadata", {}).get("source", source),
                                "multimodal": True,
                                "page": fc.get("metadata", {}).get("page", 0),
                                "type": "pdf",
                                "has_tables": "TABLE" in fc.get("content", "").upper(),
                            },
                        })

                    logger.info(
                        f"Multimodal fusion: {table_count} tables, "
                        f"{image_count} images, {len(fused_chunks)} fused chunks"
                    )
                except Exception as e:
                    logger.warning(f"Multimodal fusion failed (continuing): {e}")

            # 3. Generate document summary
            doc_summary = self.doc_processor.get_document_summary(pages)

            # 4. Chunk
            chunks = self.doc_processor.chunk_documents(pages)
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks generated from document",
                }

            # 5. Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_manager.embed_texts(texts)

            # 6. Store in vector store
            metadatas = [chunk.metadata for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            self.vector_store.add_documents(
                embeddings=embeddings,
                contents=texts,
                metadatas=metadatas,
                chunk_ids=chunk_ids,
            )

            # 7. Save index to disk
            self.vector_store.save_index()

            # 8. Refresh BM25 index
            self.retriever.refresh_bm25()

            elapsed = time.time() - start_time
            self._stats["documents_processed"] += 1
            self._stats["total_chunks_stored"] = len(
                self.vector_store.content_store
            )

            # 9. Register document in persistent registry
            try:
                import hashlib
                from pathlib import Path

                fpath = Path(file_path)
                file_size = fpath.stat().st_size if fpath.exists() else 0

                self.document_registry.register(
                    filename=doc_summary["source"],
                    metadata_dict={
                        "file_hash": doc_summary.get("file_hash", ""),
                        "file_size_bytes": file_size,
                        "file_type": fpath.suffix.lstrip(".").lower(),
                        "total_pages": doc_summary.get("total_pages", 0),
                        "chunk_count": len(chunks),
                        "has_tables": doc_summary.get("total_tables", 0) > 0,
                        "has_images": image_count > 0,
                        "processing_time_seconds": round(elapsed, 2),
                    },
                )
            except Exception as e:
                logger.warning(f"Document registry update failed: {e}")

            result = {
                "success": True,
                "document_summary": doc_summary,
                "chunks_created": len(chunks),
                "total_chunks_in_store": len(self.vector_store.content_store),
                "processing_time_seconds": round(elapsed, 2),
                "source": doc_summary["source"],
                "table_count": table_count,
                "image_count": image_count,
            }

            logger.info(
                f"Document ingested successfully: {doc_summary['source']} "
                f"({len(chunks)} chunks in {elapsed:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def query(
        self,
        question: str,
        session_id: str = "default",
        k: int = 5,
        filter_source: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.

        Pipeline: Cache Check → Retrieve → Re-rank → Augment → Generate → Cache → Cite
        """
        start_time = time.time()
        self._stats["total_queries"] += 1

        logger.info(
            f"Processing query: '{question[:80]}...' "
            f"(session={session_id}, k={k})"
        )

        try:
            # 0. Generate query embedding (used for both cache and retrieval)
            query_embedding = self.embedding_manager.embed_query(question)

            # 1. Check semantic cache
            cached = self.semantic_cache.get(question, query_embedding)
            if cached is not None:
                logger.info(f"Semantic cache hit for: '{question[:50]}...'")
                cached["metadata"]["cache_hit"] = True
                cached["metadata"]["query_time_seconds"] = round(
                    time.time() - start_time, 2
                )
                cached["metadata"]["session_id"] = session_id

                # Still save to conversation store
                self._update_conversation(
                    session_id, question, cached["answer"],
                    cached.get("sources", []), cached.get("metadata", {}),
                )
                return cached

            # 2. Build conversation context
            conv_context = self._get_conversation_context(session_id)

            # 3. Retrieve relevant chunks
            filter_metadata = None
            if filter_source:
                filter_metadata = {"source": filter_source}

            retrieved = self.retriever.retrieve(
                query=question,
                k=k * 2,  # Retrieve more for re-ranking
                filter_metadata=filter_metadata,
                use_hybrid=use_hybrid,
            )

            if not retrieved:
                return {
                    "answer": "No relevant information found in the uploaded documents. "
                              "Please try a different question or upload more documents.",
                    "sources": [],
                    "metadata": {
                        "provider": "none",
                        "retrieval_count": 0,
                        "query_time_seconds": round(time.time() - start_time, 2),
                        "cache_hit": False,
                    },
                }

            # 4. Re-rank results
            reranked = self.reranker.rerank(question, retrieved, top_k=k)

            # 5. Generate response
            result = self.generator.generate(
                query=question,
                retrieved_chunks=reranked,
                conversation_context=conv_context,
            )

            # 6. Update conversation memory (both in-memory and persistent)
            self._update_conversation(
                session_id, question, result["answer"],
                result.get("sources", []), result.get("metadata", {}),
            )

            elapsed = time.time() - start_time
            self._update_avg_query_time(elapsed)

            result["metadata"]["query_time_seconds"] = round(elapsed, 2)
            result["metadata"]["session_id"] = session_id
            result["metadata"]["cache_hit"] = False

            # 7. Cache the response
            try:
                self.semantic_cache.set(question, query_embedding, result)
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")

            logger.info(
                f"Query answered in {elapsed:.2f}s "
                f"(provider={result['metadata'].get('provider')}, "
                f"chunks={len(reranked)})"
            )

            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "query_time_seconds": round(time.time() - start_time, 2),
                    "cache_hit": False,
                },
            }

    def _get_conversation_context(self, session_id: str) -> str:
        """Get recent conversation context for a session."""
        # Try in-memory first for speed
        if session_id in self.conversations:
            history = self.conversations[session_id]
        else:
            # Fall back to persistent store
            history = self.conversation_store.get_history(session_id, limit=3)

        recent = history[-3:] if history else []

        parts = []
        for msg in recent:
            q = msg.get("question", "")
            a = msg.get("answer", "")
            parts.append(f"Q: {q}\nA: {a[:200]}")

        return "\n".join(parts)

    def _update_conversation(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ):
        """Update conversation history in both memory and persistent store."""
        # In-memory cache
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append(
            {
                "question": question,
                "answer": answer,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Keep only last 20 exchanges per session in memory
        if len(self.conversations[session_id]) > 20:
            self.conversations[session_id] = self.conversations[session_id][-20:]

        # Persistent store
        try:
            self.conversation_store.save_exchange(
                session_id=session_id,
                question=question,
                answer=answer,
                sources=sources,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to persist conversation: {e}")

    def _update_avg_query_time(self, elapsed: float):
        """Update average query time."""
        n = self._stats["total_queries"]
        current_avg = self._stats["avg_query_time"]
        self._stats["avg_query_time"] = ((current_avg * (n - 1)) + elapsed) / n

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session from persistent store."""
        return self.conversation_store.get_history(session_id)

    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
        self.conversation_store.delete_session(session_id)

    def delete_document(self, source_name: str) -> Dict:
        """Delete a document from the system."""
        self.vector_store.delete_by_source(
            source_name, embedding_manager=self.embedding_manager
        )
        self.retriever.refresh_bm25()

        # Soft-delete from registry
        try:
            self.document_registry.delete(source_name)
        except Exception as e:
            logger.warning(f"Document registry delete failed: {e}")

        return {
            "success": True,
            "message": f"Document '{source_name}' deleted",
            "remaining_chunks": len(self.vector_store.content_store),
        }

    def get_documents(self) -> List[Dict]:
        """List all ingested documents from persistent registry."""
        # Primary: document registry
        try:
            registry_docs = self.document_registry.list_documents()
            if registry_docs:
                return registry_docs
        except Exception as e:
            logger.warning(f"Document registry read failed: {e}")

        # Fallback: vector store metadata
        sources = {}
        for meta in self.vector_store.metadata_store:
            source = meta.get("source", "Unknown")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "pages": set(),
                    "chunks": 0,
                    "has_tables": False,
                    "type": meta.get("type", "unknown"),
                }
            sources[source]["pages"].add(meta.get("page", 0))
            sources[source]["chunks"] += 1
            if meta.get("has_tables"):
                sources[source]["has_tables"] = True

        docs = []
        for doc_info in sources.values():
            doc_info["total_pages"] = len(doc_info["pages"])
            doc_info["pages"] = sorted(doc_info["pages"])
            docs.append(doc_info)

        return docs

    def get_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        vs_stats = self.vector_store.get_stats()
        llm_stats = self.llm_router.get_stats()
        cache_stats = self.semantic_cache.get_stats()

        return {
            "engine": self._stats,
            "vector_store": vs_stats,
            "llm": llm_stats,
            "cache": cache_stats,
            "documents": len(self.get_documents()),
            "active_sessions": len(self.conversations),
        }
