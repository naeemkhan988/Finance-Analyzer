"""
Run Pipeline Script
===================
Interactive CLI for querying the RAG pipeline.

Usage:
    python scripts/run_pipeline.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag.pipeline import RAGPipeline
from src.utils.logger import setup_logging


def main():
    setup_logging(log_level="INFO")

    print("=" * 60)
    print("🏦 Multimodal RAG Financial Document Analyzer")
    print("=" * 60)
    print()

    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    stats = pipeline.get_stats()
    print(f"📦 Loaded {stats['vector_store']['total_chunks']} chunks")
    print(f"🤖 LLM: {stats['llm'].get('active_provider', 'fallback')}")
    print()
    print("Type your questions (type 'quit' to exit, 'stats' for system info)")
    print("-" * 60)

    session_id = "cli_session"

    while True:
        try:
            question = input("\n❓ You: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 👋")
                break

            if question.lower() == "stats":
                stats = pipeline.get_stats()
                print(f"\n📊 System Stats:")
                print(f"   Documents: {stats.get('documents', 0)}")
                print(f"   Chunks: {stats['vector_store']['total_chunks']}")
                print(f"   Queries: {stats['engine']['total_queries']}")
                print(f"   Avg Query Time: {stats['engine']['avg_query_time']:.2f}s")
                continue

            if question.lower() == "docs":
                docs = pipeline.get_documents()
                print(f"\n📄 Documents ({len(docs)}):")
                for doc in docs:
                    print(f"   - {doc['source']} ({doc['chunks']} chunks, {doc['total_pages']} pages)")
                continue

            # Query pipeline
            result = pipeline.query(
                question=question,
                session_id=session_id,
                k=5,
            )

            print(f"\n💡 Answer:\n{result['answer']}")

            if result.get("sources"):
                print(f"\n📚 Sources:")
                for src in result["sources"]:
                    print(f"   - {src['document']} (Page {src['page']}, {src['relevance_score']:.0%})")

            meta = result.get("metadata", {})
            print(f"\n⚡ {meta.get('provider', 'N/A')} | {meta.get('query_time_seconds', 0):.2f}s")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
