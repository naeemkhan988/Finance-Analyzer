"""
RAG Pipeline
=============
Full end-to-end RAG pipeline orchestrator.
Coordinates document processing, retrieval, and generation.
"""

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

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production RAG pipeline orchestrating the full workflow.

    Pipeline:
    1. Document Upload → Parse → Chunk → Embed → Store
    2. Query → Expand → Hybrid Retrieve → Re-rank → Generate → Cite

    Features:
    - End-to-end document processing pipeline
    - Hybrid retrieval (vector + keyword)
    - Multi-provider LLM generation with failover
    - Source citation and provenance tracking
    - Conversation memory for follow-up questions
    - Document management (add, delete, list)
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

        # Conversation memory: session_id -> list of messages
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

        Pipeline: Load → Parse → Chunk → Embed → Store
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

            # 2. Generate document summary
            doc_summary = self.doc_processor.get_document_summary(pages)

            # 3. Chunk
            chunks = self.doc_processor.chunk_documents(pages)
            if not chunks:
                return {
                    "success": False,
                    "error": "No chunks generated from document",
                }

            # 4. Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_manager.embed_texts(texts)

            # 5. Store in vector store
            metadatas = [chunk.metadata for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            self.vector_store.add_documents(
                embeddings=embeddings,
                contents=texts,
                metadatas=metadatas,
                chunk_ids=chunk_ids,
            )

            # 6. Save index to disk
            self.vector_store.save_index()

            # 7. Refresh BM25 index
            self.retriever.refresh_bm25()

            elapsed = time.time() - start_time
            self._stats["documents_processed"] += 1
            self._stats["total_chunks_stored"] = len(
                self.vector_store.content_store
            )

            result = {
                "success": True,
                "document_summary": doc_summary,
                "chunks_created": len(chunks),
                "total_chunks_in_store": len(self.vector_store.content_store),
                "processing_time_seconds": round(elapsed, 2),
                "source": doc_summary["source"],
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

        Pipeline: Query → Retrieve → Re-rank → Augment → Generate → Cite
        """
        start_time = time.time()
        self._stats["total_queries"] += 1

        logger.info(
            f"Processing query: '{question[:80]}...' "
            f"(session={session_id}, k={k})"
        )

        try:
            # 1. Build conversation context
            conv_context = self._get_conversation_context(session_id)

            # 2. Retrieve relevant chunks
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
                    },
                }

            # 3. Re-rank results
            reranked = self.reranker.rerank(question, retrieved, top_k=k)

            # 4. Generate response
            result = self.generator.generate(
                query=question,
                retrieved_chunks=reranked,
                conversation_context=conv_context,
            )

            # 5. Update conversation memory
            self._update_conversation(session_id, question, result["answer"])

            elapsed = time.time() - start_time
            self._update_avg_query_time(elapsed)

            result["metadata"]["query_time_seconds"] = round(elapsed, 2)
            result["metadata"]["session_id"] = session_id

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
                },
            }

    def _get_conversation_context(self, session_id: str) -> str:
        """Get recent conversation context for a session."""
        if session_id not in self.conversations:
            return ""

        history = self.conversations[session_id]
        recent = history[-3:]

        parts = []
        for msg in recent:
            parts.append(f"Q: {msg['question']}\nA: {msg['answer'][:200]}")

        return "\n".join(parts)

    def _update_conversation(self, session_id: str, question: str, answer: str):
        """Update conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append(
            {
                "question": question,
                "answer": answer,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Keep only last 20 exchanges per session
        if len(self.conversations[session_id]) > 20:
            self.conversations[session_id] = self.conversations[session_id][-20:]

    def _update_avg_query_time(self, elapsed: float):
        """Update average query time."""
        n = self._stats["total_queries"]
        current_avg = self._stats["avg_query_time"]
        self._stats["avg_query_time"] = ((current_avg * (n - 1)) + elapsed) / n

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        return self.conversations.get(session_id, [])

    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]

    def delete_document(self, source_name: str) -> Dict:
        """Delete a document from the system."""
        self.vector_store.delete_by_source(source_name)
        self.retriever.refresh_bm25()
        return {
            "success": True,
            "message": f"Document '{source_name}' deleted",
            "remaining_chunks": len(self.vector_store.content_store),
        }

    def get_documents(self) -> List[Dict]:
        """List all ingested documents."""
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

        return {
            "engine": self._stats,
            "vector_store": vs_stats,
            "llm": llm_stats,
            "documents": len(self.get_documents()),
            "active_sessions": len(self.conversations),
        }
