"""
Multimodal Processing Module
============================
Handles extraction and analysis of images, tables, and OCR from documents.
"""

from src.multimodal.image_extractor import ImageExtractor
from src.multimodal.table_extractor import TableExtractor
from src.multimodal.ocr_processor import OCRProcessor
from src.multimodal.multimodal_fusion import MultimodalFusion

__all__ = [
    "ImageExtractor",
    "TableExtractor",
    "OCRProcessor",
    "MultimodalFusion",
]
