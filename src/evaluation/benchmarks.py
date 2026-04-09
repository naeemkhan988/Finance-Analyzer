"""
Benchmarks
==========
Pre-defined benchmark test suites for financial RAG evaluation.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


# Pre-defined financial document benchmark questions
FINANCIAL_QA_BENCHMARKS = [
    {
        "question": "What was the total revenue for the most recent fiscal year?",
        "category": "financial_metrics",
        "difficulty": "easy",
        "expected_entities": ["revenue", "fiscal year"],
    },
    {
        "question": "How did net income compare between Q3 and Q4?",
        "category": "comparison",
        "difficulty": "medium",
        "expected_entities": ["net income", "Q3", "Q4"],
    },
    {
        "question": "What are the main risk factors mentioned in the annual report?",
        "category": "risk_analysis",
        "difficulty": "hard",
        "expected_entities": ["risk factors"],
    },
    {
        "question": "What is the company's debt-to-equity ratio?",
        "category": "financial_ratios",
        "difficulty": "medium",
        "expected_entities": ["debt", "equity", "ratio"],
    },
    {
        "question": "Summarize the key changes in operating expenses year-over-year.",
        "category": "trend_analysis",
        "difficulty": "hard",
        "expected_entities": ["operating expenses", "year-over-year"],
    },
    {
        "question": "What dividend was declared per share?",
        "category": "shareholder_info",
        "difficulty": "easy",
        "expected_entities": ["dividend", "per share"],
    },
    {
        "question": "What is the gross profit margin trend over the last 3 quarters?",
        "category": "trend_analysis",
        "difficulty": "hard",
        "expected_entities": ["gross profit margin", "quarters"],
    },
    {
        "question": "What acquisitions or investments were made during the year?",
        "category": "corporate_actions",
        "difficulty": "medium",
        "expected_entities": ["acquisitions", "investments"],
    },
]


class BenchmarkSuite:
    """
    Pre-defined benchmark suites for evaluating RAG quality.

    Suites:
    - financial_qa: General financial Q&A
    - retrieval_stress: Tests retrieval accuracy under different conditions
    - edge_cases: Boundary and adversarial test cases
    """

    @staticmethod
    def get_financial_qa_suite() -> List[Dict]:
        """Get the financial Q&A benchmark suite."""
        return FINANCIAL_QA_BENCHMARKS

    @staticmethod
    def get_retrieval_stress_suite() -> List[Dict]:
        """Get retrieval stress test cases."""
        return [
            {
                "question": "revenue",
                "category": "single_word",
                "difficulty": "test",
                "description": "Single word query - should still retrieve relevant results",
            },
            {
                "question": "What is the impact of foreign exchange rate fluctuations on consolidated revenue, "
                           "particularly in the Asia-Pacific region during the fourth fiscal quarter?",
                "category": "complex_query",
                "difficulty": "test",
                "description": "Very specific, long query",
            },
            {
                "question": "Tell me about the thing with the money",
                "category": "vague_query",
                "difficulty": "test",
                "description": "Intentionally vague query",
            },
        ]

    @staticmethod
    def get_edge_cases_suite() -> List[Dict]:
        """Get edge case test scenarios."""
        return [
            {
                "question": "",
                "category": "empty_query",
                "expected_behavior": "Should return validation error",
            },
            {
                "question": "What is 2+2?",
                "category": "off_topic",
                "expected_behavior": "Should indicate no relevant documents found",
            },
            {
                "question": "SELECT * FROM users; DROP TABLE;",
                "category": "injection_test",
                "expected_behavior": "Should handle safely without issues",
            },
        ]

    @staticmethod
    def get_all_suites() -> Dict[str, List[Dict]]:
        """Get all benchmark suites."""
        return {
            "financial_qa": BenchmarkSuite.get_financial_qa_suite(),
            "retrieval_stress": BenchmarkSuite.get_retrieval_stress_suite(),
            "edge_cases": BenchmarkSuite.get_edge_cases_suite(),
        }
