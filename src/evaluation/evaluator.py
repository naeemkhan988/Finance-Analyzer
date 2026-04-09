"""
RAG Evaluator
=============
End-to-end evaluation of the RAG pipeline.
"""

import json
import logging
import time
from typing import List, Dict, Optional
from pathlib import Path

from src.evaluation.metrics import RAGMetrics

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using test datasets.

    Workflow:
    1. Load evaluation dataset (questions + expected answers/sources)
    2. Run queries through pipeline
    3. Compute retrieval and generation metrics
    4. Generate evaluation report
    """

    def __init__(self, rag_pipeline=None):
        self.pipeline = rag_pipeline
        self.metrics = RAGMetrics()
        self._results: List[Dict] = []

    def load_test_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load evaluation dataset from JSON.

        Expected format:
        [
            {
                "question": "What was the revenue in Q3?",
                "expected_answer": "Revenue was $5.2 billion",
                "relevant_sources": ["annual_report.pdf"],
                "relevant_pages": [12, 13]
            }
        ]
        """
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} test cases from {dataset_path}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load test dataset: {e}")
            return []

    def evaluate(
        self,
        test_cases: List[Dict],
        k: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        Run full evaluation across test cases.

        Returns:
            Aggregate metrics and per-query results
        """
        if not self.pipeline:
            return {"error": "RAG pipeline not initialized"}

        self._results = []
        total_time = 0

        for i, test_case in enumerate(test_cases):
            question = test_case["question"]
            expected = test_case.get("expected_answer", "")
            relevant_sources = test_case.get("relevant_sources", [])

            start = time.time()
            result = self.pipeline.query(question=question, k=k)
            elapsed = time.time() - start
            total_time += elapsed

            # Compute metrics
            retrieved_sources = [
                s.get("document", "") for s in result.get("sources", [])
            ]

            eval_result = {
                "question": question,
                "answer": result.get("answer", ""),
                "expected_answer": expected,
                "retrieved_sources": retrieved_sources,
                "relevant_sources": relevant_sources,
                "latency_seconds": round(elapsed, 2),
                "metrics": {
                    "precision_at_k": self.metrics.precision_at_k(
                        retrieved_sources, relevant_sources, k
                    ),
                    "recall_at_k": self.metrics.recall_at_k(
                        retrieved_sources, relevant_sources, k
                    ),
                    "mrr": self.metrics.mrr(retrieved_sources, relevant_sources),
                    "answer_relevance": self.metrics.answer_relevance_score(
                        result.get("answer", ""), question
                    ),
                    "faithfulness": self.metrics.faithfulness_score(
                        result.get("answer", ""),
                        " ".join(s.get("chunk_preview", "") for s in result.get("sources", [])),
                    ),
                },
            }

            self._results.append(eval_result)

            if verbose:
                logger.info(
                    f"[{i+1}/{len(test_cases)}] "
                    f"P@{k}={eval_result['metrics']['precision_at_k']:.2f} "
                    f"R@{k}={eval_result['metrics']['recall_at_k']:.2f} "
                    f"MRR={eval_result['metrics']['mrr']:.2f} "
                    f"({elapsed:.2f}s)"
                )

        # Aggregate metrics
        return self._aggregate_results(total_time)

    def _aggregate_results(self, total_time: float) -> Dict:
        """Compute aggregate metrics across all test cases."""
        if not self._results:
            return {"error": "No results to aggregate"}

        n = len(self._results)

        avg_metrics = {}
        metric_keys = self._results[0]["metrics"].keys()
        for key in metric_keys:
            values = [r["metrics"][key] for r in self._results]
            avg_metrics[f"avg_{key}"] = round(sum(values) / n, 4)

        return {
            "total_test_cases": n,
            "total_time_seconds": round(total_time, 2),
            "avg_latency_seconds": round(total_time / n, 2),
            "aggregate_metrics": avg_metrics,
            "per_query_results": self._results,
        }

    def save_report(self, output_path: str):
        """Save evaluation report to JSON."""
        report = self._aggregate_results(
            sum(r["latency_seconds"] for r in self._results)
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to {output_path}")
