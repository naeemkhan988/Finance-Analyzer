"""
Evaluation Metrics
==================
Metrics for evaluating RAG pipeline quality.
"""

import logging
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class RAGMetrics:
    """
    Metrics for evaluating RAG pipeline performance.

    Metrics:
    - Retrieval: Precision@K, Recall@K, MRR, NDCG
    - Generation: Answer relevance, faithfulness
    - Latency: Query time, retrieval time, generation time
    """

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """
        Precision@K: fraction of retrieved docs that are relevant.

        Args:
            retrieved: List of retrieved document IDs
            relevant: List of ground-truth relevant document IDs
            k: Number of top results to consider
        """
        if not retrieved or not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)
        hits = sum(1 for doc in top_k if doc in relevant_set)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Recall@K: fraction of relevant docs that were retrieved."""
        if not relevant:
            return 0.0

        top_k = set(retrieved[:k])
        relevant_set = set(relevant)
        hits = len(top_k & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant result."""
        relevant_set = set(relevant)
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain."""
        import math

        relevant_set = set(relevant)
        top_k = retrieved[:k]

        # DCG
        dcg = sum(
            1.0 / math.log2(i + 2) for i, doc in enumerate(top_k) if doc in relevant_set
        )

        # Ideal DCG
        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def answer_relevance_score(answer: str, query: str) -> float:
        """
        Simple keyword-based relevance score between answer and query.
        For production, use an LLM-based evaluator.
        """
        if not answer or not query:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "what", "how", "of", "in", "to"}
        query_keywords = query_words - stop_words
        answer_keywords = answer_words - stop_words

        if not query_keywords:
            return 0.0

        overlap = len(query_keywords & answer_keywords)
        return overlap / len(query_keywords)

    @staticmethod
    def faithfulness_score(answer: str, context: str) -> float:
        """
        Check if the answer is grounded in the provided context.
        Simple overlap-based metric.
        """
        if not answer or not context:
            return 0.0

        answer_sentences = answer.split(". ")
        context_lower = context.lower()

        grounded = 0
        for sentence in answer_sentences:
            # Check if key phrases from the sentence appear in context
            words = sentence.lower().split()
            if len(words) < 3:
                continue
            # Check for 3-gram overlap
            for i in range(len(words) - 2):
                trigram = " ".join(words[i:i + 3])
                if trigram in context_lower:
                    grounded += 1
                    break

        return grounded / max(len(answer_sentences), 1)
