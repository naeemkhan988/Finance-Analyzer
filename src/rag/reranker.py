"""
Re-ranker Module
================
Advanced re-ranking of retrieved documents for improved precision.
Supports cross-encoder re-ranking and heuristic-based methods.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Reranker:
    """
    Document re-ranker for improving retrieval quality.

    Strategies:
    - Cross-encoder re-ranking (when model is available)
    - Financial relevance scoring
    - Diversity-aware re-ranking (MMR)
    - Heuristic-based re-ranking (fallback)
    """

    def __init__(
        self,
        use_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        diversity_lambda: float = 0.5,
    ):
        self.use_cross_encoder = use_cross_encoder
        self.diversity_lambda = diversity_lambda
        self._cross_encoder = None

        if use_cross_encoder:
            self._load_cross_encoder(cross_encoder_model)

    def _load_cross_encoder(self, model_name: str):
        """Load cross-encoder model for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(model_name)
            logger.info(f"Cross-encoder loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Cross-encoder loading failed: {e}. Using heuristic re-ranking.")
            self.use_cross_encoder = False

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Re-rank retrieved results for better relevance ordering.

        Args:
            query: Original user query
            results: List of retrieval results to re-rank
            top_k: Number of top results to return

        Returns:
            Re-ranked list of results
        """
        if not results:
            return results

        if self.use_cross_encoder and self._cross_encoder:
            return self._cross_encoder_rerank(query, results, top_k)

        return self._heuristic_rerank(query, results, top_k)

    def _cross_encoder_rerank(
        self, query: str, results: List[Dict], top_k: int
    ) -> List[Dict]:
        """Re-rank using cross-encoder model."""
        pairs = [(query, r["content"]) for r in results]

        try:
            scores = self._cross_encoder.predict(pairs)

            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)

            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}")
            return results[:top_k]

    def _heuristic_rerank(
        self, query: str, results: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        Heuristic-based re-ranking using multiple signals.

        Signals:
        - Original retrieval score
        - Financial entity overlap
        - Table content boost
        - Query term coverage
        """
        query_terms = set(query.lower().split())

        for result in results:
            base_score = result.get("score", 0)
            content_lower = result["content"].lower()
            metadata = result.get("metadata", {})

            # Query term coverage boost
            coverage = sum(1 for term in query_terms if term in content_lower)
            coverage_score = coverage / max(len(query_terms), 1)

            # Financial entity boost
            financial_entities = metadata.get("financial_entities", {})
            entity_score = min(len(financial_entities) * 0.05, 0.2)

            # Table content boost
            table_boost = 0.1 if metadata.get("has_tables") else 0

            # Combined heuristic score
            combined = (
                base_score * 0.6
                + coverage_score * 0.25
                + entity_score * 0.1
                + table_boost * 0.05
            )

            result["rerank_score"] = combined

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]

    def mmr_rerank(
        self,
        query_embedding,
        results: List[Dict],
        embeddings,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance (MMR) re-ranking for diversity.

        Balances relevance with diversity to avoid redundant results.
        """
        import numpy as np

        if not results or len(results) <= top_k:
            return results

        selected = []
        candidates = list(range(len(results)))

        # Select first result (most relevant)
        selected.append(candidates.pop(0))

        while len(selected) < top_k and candidates:
            best_score = -float("inf")
            best_idx = -1

            for idx in candidates:
                relevance = results[idx].get("score", 0)

                # Max similarity to already selected
                max_sim = 0
                for sel_idx in selected:
                    sim = float(np.dot(embeddings[idx], embeddings[sel_idx]))
                    max_sim = max(max_sim, sim)

                mmr_score = (
                    self.diversity_lambda * relevance
                    - (1 - self.diversity_lambda) * max_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                candidates.remove(best_idx)
                selected.append(best_idx)

        return [results[i] for i in selected]
