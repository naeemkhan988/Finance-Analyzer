"""
Retriever Module
================
Hybrid retrieval with vector search + keyword BM25 re-ranking.
Implements query expansion, multi-query retrieval, and result fusion.
"""

import re
import math
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np

from src.rag.embeddings import EmbeddingManager
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class BM25:
    """
    Lightweight BM25 implementation for keyword-based retrieval.
    Used alongside vector search for hybrid retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []
        self.term_freqs: List[Dict[str, int]] = []

    def fit(self, documents: List[str]):
        """Index documents for BM25 scoring."""
        self.doc_count = len(documents)
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))

            tf = Counter(tokens)
            self.term_freqs.append(dict(tf))

            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = (
            sum(self.doc_lens) / self.doc_count if self.doc_count > 0 else 0
        )

    def score(self, query: str) -> List[float]:
        """Score all documents against a query."""
        query_tokens = self._tokenize(query)
        scores = []

        for i in range(self.doc_count):
            score = 0.0
            doc_len = self.doc_lens[i]

            for token in query_tokens:
                if token not in self.term_freqs[i]:
                    continue

                tf = self.term_freqs[i][token]
                df = self.doc_freqs.get(token, 0)

                # IDF
                idf = math.log(
                    (self.doc_count - df + 0.5) / (df + 0.5) + 1
                )

                # TF normalization
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                )

                score += idf * tf_norm

            scores.append(score)

        return scores

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with lowercase and alphanumeric filtering."""
        return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """
    Production hybrid retriever combining vector search and BM25.

    Features:
    - Semantic vector search (FAISS)
    - Keyword BM25 search
    - Reciprocal Rank Fusion (RRF) for result merging
    - Query expansion for better recall
    - Multi-query retrieval
    - Metadata filtering
    - Configurable weights for vector vs keyword search
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.bm25 = BM25()
        self._index_bm25()

    def _index_bm25(self):
        """Build BM25 index from vector store contents."""
        if self.vector_store.content_store:
            self.bm25.fit(self.vector_store.content_store)
            logger.info(
                f"BM25 index built with {len(self.vector_store.content_store)} documents"
            )

    def refresh_bm25(self):
        """Rebuild BM25 index (call after adding new documents)."""
        self._index_bm25()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_hybrid: bool = True,
        expand_query: bool = True,
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User query string
            k: Number of results to return
            filter_metadata: Optional metadata filter
            use_hybrid: Whether to use hybrid search (vector + BM25)
            expand_query: Whether to expand the query for better recall

        Returns:
            List of result dicts with content, metadata, score, and source info
        """
        if not self.vector_store.content_store:
            return []

        # Query expansion
        queries = [query]
        if expand_query:
            expanded = self._expand_query(query)
            queries.extend(expanded)

        all_results = {}

        for q in queries:
            # Vector search
            vector_results = self._vector_search(
                q, k=k * 2, filter_metadata=filter_metadata
            )

            # BM25 keyword search
            keyword_results = []
            if use_hybrid and self.bm25.doc_count > 0:
                keyword_results = self._keyword_search(q, k=k * 2)

            # Merge results using RRF
            if use_hybrid and keyword_results:
                merged = self._reciprocal_rank_fusion(
                    vector_results, keyword_results
                )
            else:
                merged = vector_results

            # Deduplicate
            for result in merged:
                chunk_id = result.get("chunk_id", "")
                if chunk_id not in all_results or result["score"] > all_results[chunk_id]["score"]:
                    all_results[chunk_id] = result

        # Sort by score and return top-k
        final = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        # Apply metadata filter post-merge if needed
        if filter_metadata:
            final = [
                r for r in final
                if self._matches_filter(r.get("metadata", {}), filter_metadata)
            ]

        results = final[:k]

        logger.info(
            f"Retrieved {len(results)} results for query: '{query[:50]}...' "
            f"(hybrid={use_hybrid}, expanded={len(queries)} queries)"
        )

        return results

    def _vector_search(
        self, query: str, k: int = 10, filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Perform semantic vector search."""
        query_embedding = self.embedding_manager.embed_query(query)
        results = self.vector_store.search(
            query_embedding, k=k, filter_metadata=filter_metadata
        )

        # Normalize scores to 0-1
        if results:
            max_score = max(r["score"] for r in results)
            min_score = min(r["score"] for r in results)
            score_range = max_score - min_score if max_score != min_score else 1

            for r in results:
                r["score"] = (r["score"] - min_score) / score_range
                r["search_type"] = "vector"

        return results

    def _keyword_search(self, query: str, k: int = 10) -> List[Dict]:
        """Perform BM25 keyword search."""
        scores = self.bm25.score(query)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        max_score = max(scores) if scores else 1
        min_score = min(s for _, s in indexed_scores[:k]) if indexed_scores else 0
        score_range = max_score - min_score if max_score != min_score else 1

        for idx, score in indexed_scores[:k]:
            if score <= 0:
                continue
            results.append(
                {
                    "content": self.vector_store.content_store[idx],
                    "metadata": self.vector_store.metadata_store[idx],
                    "score": (score - min_score) / score_range if score_range > 0 else 0,
                    "chunk_id": self.vector_store.chunk_ids[idx],
                    "search_type": "keyword",
                }
            )

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
    ) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        RRF score = sum(1 / (k + rank)) across all result lists.
        """
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict] = {}

        for rank, result in enumerate(vector_results):
            chunk_id = result.get("chunk_id", str(rank))
            rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
            result_map[chunk_id] = result

        for rank, result in enumerate(keyword_results):
            chunk_id = result.get("chunk_id", f"kw_{rank}")
            rrf_score = self.keyword_weight / (self.rrf_k + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in result_map:
                result_map[chunk_id] = result

        for chunk_id, score in rrf_scores.items():
            if chunk_id in result_map:
                result_map[chunk_id]["score"] = score
                result_map[chunk_id]["search_type"] = "hybrid"

        return list(result_map.values())

    def _expand_query(self, query: str) -> List[str]:
        """Simple query expansion using financial synonyms and reformulations."""
        expansions = []

        financial_synonyms = {
            "revenue": ["sales", "top-line", "income"],
            "profit": ["net income", "earnings", "bottom-line"],
            "growth": ["increase", "YoY change", "expansion"],
            "debt": ["liabilities", "borrowings", "leverage"],
            "margin": ["profitability", "gross margin", "operating margin"],
            "eps": ["earnings per share", "EPS"],
            "roi": ["return on investment", "ROI"],
            "ebitda": ["EBITDA", "operating earnings"],
            "cash flow": ["free cash flow", "FCF", "operating cash flow"],
            "market cap": ["market capitalization", "valuation"],
        }

        query_lower = query.lower()
        for term, synonyms in financial_synonyms.items():
            if term in query_lower:
                for syn in synonyms[:1]:
                    expanded = query_lower.replace(term, syn)
                    expansions.append(expanded)
                break

        return expansions[:2]

    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
