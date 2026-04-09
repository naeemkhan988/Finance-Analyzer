"""
Multimodal Tests
================
Tests for multimodal processing (tables, images, OCR).
"""

import pytest


class TestTableExtractor:
    """Test table extraction."""

    def test_process_table(self):
        """Test table data processing."""
        from src.multimodal.table_extractor import TableExtractor

        extractor = TableExtractor()
        table_data = [
            ["Metric", "Q1", "Q2", "Q3"],
            ["Revenue", "$1M", "$1.5M", "$2M"],
            ["Profit", "$100K", "$200K", "$300K"],
        ]

        result = extractor._process_table(table_data, "test.pdf", 1, 0)

        assert result is not None
        assert result["headers"] == ["Metric", "Q1", "Q2", "Q3"]
        assert result["row_count"] == 2
        assert result["is_financial"] is True

    def test_is_financial_table(self):
        """Test financial table detection."""
        import pandas as pd
        from src.multimodal.table_extractor import TableExtractor

        extractor = TableExtractor()

        # Financial table
        df = pd.DataFrame({"Revenue": [100], "Expenses": [50]})
        assert extractor._is_financial_table(df) is True

        # Non-financial table
        df2 = pd.DataFrame({"Name": ["Alice"], "Age": [30]})
        assert extractor._is_financial_table(df2) is False

    def test_tables_to_chunks(self):
        """Test table to chunk conversion."""
        from src.multimodal.table_extractor import TableExtractor

        extractor = TableExtractor()
        tables = [
            {
                "text_representation": "Col1 | Col2\n---\nA | B",
                "source": "test.pdf",
                "page": 1,
                "is_financial": True,
                "row_count": 1,
                "column_count": 2,
            }
        ]

        chunks = extractor.tables_to_chunks(tables)
        assert len(chunks) == 1
        assert "[TABLE]" in chunks[0]["content"]


class TestImageExtractor:
    """Test image extraction."""

    def test_classify_image(self):
        """Test image classification by dimensions."""
        from src.multimodal.image_extractor import ImageExtractor

        extractor = ImageExtractor()

        assert extractor._classify_image(50, 50) == "icon_or_logo"
        assert extractor._classify_image(800, 400) == "chart_or_graph"
        assert extractor._classify_image(500, 30) == "signature_or_banner"

    def test_get_image_stats(self):
        """Test image statistics."""
        from src.multimodal.image_extractor import ImageExtractor

        extractor = ImageExtractor()
        images = [
            {"page": 1, "type": "chart_or_graph"},
            {"page": 1, "type": "icon_or_logo"},
            {"page": 2, "type": "chart_or_graph"},
        ]

        stats = extractor.get_image_stats(images)
        assert stats["total_images"] == 3
        assert stats["chart_count"] == 2
        assert stats["pages_with_images"] == 2


class TestOCRProcessor:
    """Test OCR processing."""

    def test_init(self):
        """Test OCR processor initialization."""
        from src.multimodal.ocr_processor import OCRProcessor

        processor = OCRProcessor()
        # Tesseract may or may not be available
        assert isinstance(processor._tesseract_available, bool)
