"""
Evaluation Script
=================
Run evaluation benchmarks on the RAG pipeline.

Usage:
    python scripts/evaluate.py --dataset ./data/evaluation/test_set.json
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag.pipeline import RAGPipeline
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.benchmarks import BenchmarkSuite
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--dataset", "-d", default=None, help="Path to evaluation dataset JSON")
    parser.add_argument("--suite", "-s", default="financial_qa",
                        choices=["financial_qa", "retrieval_stress", "edge_cases"],
                        help="Benchmark suite to run")
    parser.add_argument("--output", "-o", default="./data/evaluation/results.json",
                        help="Output path for evaluation report")
    parser.add_argument("--k", type=int, default=5, help="Top-K for retrieval")
    args = parser.parse_args()

    setup_logging(log_level="INFO")

    logger.info("Initializing RAG pipeline for evaluation...")
    pipeline = RAGPipeline()

    evaluator = RAGEvaluator(rag_pipeline=pipeline)

    # Load test cases
    if args.dataset:
        test_cases = evaluator.load_test_dataset(args.dataset)
    else:
        logger.info(f"Using benchmark suite: {args.suite}")
        suites = BenchmarkSuite.get_all_suites()
        test_cases = suites.get(args.suite, [])

    if not test_cases:
        logger.error("No test cases found!")
        return

    logger.info(f"Running {len(test_cases)} test cases...")

    # Run evaluation
    results = evaluator.evaluate(test_cases, k=args.k, verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total test cases: {results.get('total_test_cases', 0)}")
    print(f"Total time: {results.get('total_time_seconds', 0):.2f}s")
    print(f"Avg latency: {results.get('avg_latency_seconds', 0):.2f}s")

    metrics = results.get("aggregate_metrics", {})
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save report
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(args.output)
    print(f"\n📄 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
