"""
Response Generator
==================
LLM-based response generation with context augmentation and citation.
"""

import logging
import time
from typing import Dict, List, Optional

from src.models.llm import LLMRouter

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates LLM responses augmented with retrieved document context.

    Features:
    - Context-aware prompt construction
    - Source citation extraction
    - Multi-provider LLM support
    - Conversation memory integration
    - Structured output formatting
    """

    def __init__(self, llm_router: Optional[LLMRouter] = None):
        self.llm_router = llm_router or LLMRouter()

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        conversation_context: str = "",
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Dict:
        """
        Generate a response using retrieved context.

        Args:
            query: User's question
            retrieved_chunks: List of retrieval results
            conversation_context: Previous conversation context
            system_prompt: Custom system prompt
            max_tokens: Max response tokens
            temperature: Response creativity level

        Returns:
            Dict with answer, sources, and metadata
        """
        start_time = time.time()

        # Build augmented context
        context = self._build_context(retrieved_chunks)

        # Enhance query if conversation context exists
        enhanced_query = query
        if conversation_context:
            enhanced_query = (
                f"Previous context: {conversation_context}\n\n"
                f"Current question: {query}"
            )

        # Generate via LLM
        llm_result = self.llm_router.generate(
            query=enhanced_query,
            context=context,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract source citations
        sources = self._extract_sources(retrieved_chunks)

        elapsed = time.time() - start_time

        return {
            "answer": llm_result["answer"],
            "sources": sources,
            "metadata": {
                "provider": llm_result["provider"],
                "model": llm_result["model"],
                "latency_seconds": llm_result["latency_seconds"],
                "total_time_seconds": round(elapsed, 2),
                "retrieval_count": len(retrieved_chunks),
                "tokens_used": llm_result["tokens_used"],
                "finish_reason": llm_result["finish_reason"],
            },
        }

    def _build_context(self, retrieved: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, result in enumerate(retrieved, 1):
            source = result.get("metadata", {}).get("source", "Unknown")
            page = result.get("metadata", {}).get("page", "?")
            score = result.get("score", 0)

            context_parts.append(
                f"[Source {i}: {source}, Page {page} (relevance: {score:.2f})]\n"
                f"{result['content']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, retrieved: List[Dict]) -> List[Dict]:
        """Extract clean source citations from retrieved chunks."""
        sources = []
        seen = set()

        for result in retrieved:
            meta = result.get("metadata", {})
            source_key = f"{meta.get('source', 'Unknown')}_p{meta.get('page', '?')}"

            if source_key not in seen:
                seen.add(source_key)
                sources.append(
                    {
                        "document": meta.get("source", "Unknown"),
                        "page": meta.get("page", "?"),
                        "relevance_score": round(result.get("score", 0), 3),
                        "has_tables": meta.get("has_tables", False),
                        "chunk_preview": result["content"][:200] + "..."
                        if len(result["content"]) > 200
                        else result["content"],
                    }
                )

        return sources
