"""
Data Ingestion Script
=====================
Batch ingest documents from a directory into the RAG pipeline.

Usage:
    python scripts/ingest_data.py --input ./data/raw --verbose
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.rag.pipeline import RAGPipeline
from src.utils.logger import setup_logging
from src.utils.file_utils import list_files

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG pipeline")
    parser.add_argument("--input", "-i", default="./data/raw", help="Input directory")
    parser.add_argument("--extensions", "-e", nargs="+", default=["pdf", "txt", "docx"],
                        help="File extensions to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level, log_file="./logs/ingestion.log")

    logger.info(f"Starting batch ingestion from: {args.input}")

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Find documents
    files = list_files(args.input, extensions=args.extensions)
    logger.info(f"Found {len(files)} documents to process")

    if not files:
        logger.warning("No documents found. Check the input directory.")
        return

    # Process each document
    success_count = 0
    error_count = 0

    for i, file_path in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing: {file_path.name}")

        result = pipeline.ingest_document(str(file_path))

        if result.get("success"):
            success_count += 1
            logger.info(
                f"  ✅ {result['chunks_created']} chunks "
                f"({result['processing_time_seconds']:.1f}s)"
            )
        else:
            error_count += 1
            logger.error(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Ingestion Complete!")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failed:  {error_count}")
    logger.info(f"  📦 Total chunks: {pipeline.vector_store.get_stats()['total_chunks']}")


if __name__ == "__main__":
    main()
