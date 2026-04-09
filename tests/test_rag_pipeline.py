"""
RAG Pipeline Tests
==================
End-to-end tests for the RAG pipeline.
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestDocumentProcessor:
    """Test document loading and chunking."""

    def test_load_text_file(self):
        """Test loading a plain text file."""
        from src.rag.document_loader import DocumentProcessor

        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        # Create temp text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test financial document. Revenue was $1M in Q3 2024.")
            temp_path = f.name

        try:
            pages = processor.load_document(temp_path)
            assert len(pages) > 0
            assert pages[0]["content"]
            assert pages[0]["metadata"]["type"] == "txt"
        finally:
            os.unlink(temp_path)

    def test_chunk_documents(self):
        """Test document chunking."""
        from src.rag.document_loader import DocumentProcessor

        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

        pages = [
            {
                "content": "This is a long text " * 20,
                "metadata": {"source": "test.txt", "page": 1, "type": "txt"},
            }
        ]

        chunks = processor.chunk_documents(pages)
        assert len(chunks) > 1
        assert all(chunk.content for chunk in chunks)

    def test_financial_entity_detection(self):
        """Test financial entity detection."""
        from src.rag.document_loader import DocumentProcessor

        processor = DocumentProcessor()
        text = "Revenue was $5.2 billion in Q3 2024, up 15% YoY. AAPL stock price rose."

        entities = processor._detect_financial_entities(text)
        assert "currency" in entities or "percentage" in entities

    def test_document_summary(self):
        """Test document summary generation."""
        from src.rag.document_loader import DocumentProcessor

        processor = DocumentProcessor()
        pages = [
            {
                "content": "Revenue was $1M. Profit was 10%.",
                "metadata": {"source": "test.pdf", "page": 1, "table_count": 0},
            }
        ]

        summary = processor.get_document_summary(pages)
        assert summary["total_pages"] == 1
        assert summary["source"] == "test.pdf"


class TestTextSplitter:
    """Test text splitting strategies."""

    def test_basic_split(self):
        """Test basic text splitting."""
        from src.rag.text_splitter import TextSplitter, SplitConfig

        splitter = TextSplitter(SplitConfig(chunk_size=50, chunk_overlap=10))
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."

        chunks = splitter.split_text(text)
        assert len(chunks) > 0
        assert all(len(c) <= 60 for c in chunks)  # Allow some overlap

    def test_empty_text(self):
        """Test splitting empty text."""
        from src.rag.text_splitter import TextSplitter

        splitter = TextSplitter()
        chunks = splitter.split_text("")
        assert len(chunks) == 0

    def test_short_text(self):
        """Test splitting text shorter than chunk size."""
        from src.rag.text_splitter import TextSplitter

        splitter = TextSplitter()
        chunks = splitter.split_text("Short text")
        assert len(chunks) == 1
