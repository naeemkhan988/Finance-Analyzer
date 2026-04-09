"""
Multimodal Fusion
=================
Combines text, table, and image data into unified representations
for enhanced RAG retrieval and analysis.
"""

import logging
from typing import List, Dict, Optional

from src.multimodal.image_extractor import ImageExtractor
from src.multimodal.table_extractor import TableExtractor
from src.multimodal.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)


class MultimodalFusion:
    """
    Fuses multimodal content (text + tables + images) for RAG.

    Features:
    - Unified document representation
    - Cross-modal context linking
    - Table-text alignment
    - Image caption integration
    - Financial content prioritization
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_images: bool = True,
        enable_tables: bool = True,
    ):
        self.enable_ocr = enable_ocr
        self.enable_images = enable_images
        self.enable_tables = enable_tables

        self.image_extractor = ImageExtractor() if enable_images else None
        self.table_extractor = TableExtractor() if enable_tables else None
        self.ocr_processor = OCRProcessor() if enable_ocr else None

    def process_document(self, file_path: str) -> Dict:
        """
        Process a document with full multimodal extraction.

        Args:
            file_path: Path to the document

        Returns:
            Dict with text, tables, images, and fused content
        """
        result = {
            "text_chunks": [],
            "tables": [],
            "images": [],
            "ocr_text": [],
            "fused_chunks": [],
            "metadata": {},
        }

        # Extract tables
        if self.enable_tables and self.table_extractor:
            try:
                if file_path.lower().endswith(".pdf"):
                    tables = self.table_extractor.extract_from_pdf(file_path)
                elif file_path.lower().endswith((".xlsx", ".xls")):
                    tables = self.table_extractor.extract_from_excel(file_path)
                elif file_path.lower().endswith(".csv"):
                    tables = self.table_extractor.extract_from_csv(file_path)
                else:
                    tables = []

                result["tables"] = tables
                # Convert tables to text chunks
                table_chunks = self.table_extractor.tables_to_chunks(tables)
                result["fused_chunks"].extend(table_chunks)

                logger.info(f"Extracted {len(tables)} tables")
            except Exception as e:
                logger.error(f"Table extraction failed: {e}")

        # Extract images
        if self.enable_images and self.image_extractor:
            try:
                if file_path.lower().endswith(".pdf"):
                    images = self.image_extractor.extract_from_pdf(file_path)
                    result["images"] = images

                    # Run OCR on chart images
                    if self.enable_ocr and self.ocr_processor:
                        for img in images:
                            if img.get("type") == "chart_or_graph" and img.get("file_path"):
                                ocr_result = self.ocr_processor.process_image(
                                    img["file_path"]
                                )
                                if ocr_result.get("text"):
                                    result["ocr_text"].append(ocr_result)
                                    result["fused_chunks"].append({
                                        "content": f"[CHART TEXT]\n{ocr_result['text']}",
                                        "metadata": {
                                            "source": img.get("source_pdf", ""),
                                            "page": img.get("page", 0),
                                            "type": "chart_ocr",
                                        },
                                    })

                    logger.info(f"Extracted {len(images)} images")
            except Exception as e:
                logger.error(f"Image extraction failed: {e}")

        # Check for scanned PDF and run full OCR
        if self.enable_ocr and self.ocr_processor and file_path.lower().endswith(".pdf"):
            try:
                if self.ocr_processor.is_scanned_pdf(file_path):
                    logger.info("Scanned PDF detected, running full OCR...")
                    ocr_result = self.ocr_processor.process_pdf_page(file_path, 0)
                    if ocr_result.get("text"):
                        result["ocr_text"].append(ocr_result)
                        result["fused_chunks"].append({
                            "content": ocr_result["text"],
                            "metadata": {
                                "type": "ocr_full_page",
                                "page": ocr_result.get("page", 1),
                            },
                        })
            except Exception as e:
                logger.error(f"OCR processing failed: {e}")

        # Build metadata summary
        result["metadata"] = {
            "table_count": len(result["tables"]),
            "image_count": len(result["images"]),
            "ocr_segments": len(result["ocr_text"]),
            "fused_chunks": len(result["fused_chunks"]),
            "has_financial_tables": any(
                t.get("is_financial") for t in result["tables"]
            ),
        }

        return result

    def merge_with_text_chunks(
        self,
        text_chunks: List[Dict],
        multimodal_result: Dict,
    ) -> List[Dict]:
        """
        Merge multimodal content with text chunks for unified indexing.

        Args:
            text_chunks: Existing text chunks from document loader
            multimodal_result: Output from process_document()

        Returns:
            Combined list of all chunks (text + tables + OCR)
        """
        all_chunks = list(text_chunks)

        # Add fused chunks from multimodal processing
        for chunk in multimodal_result.get("fused_chunks", []):
            chunk["metadata"]["multimodal"] = True
            all_chunks.append(chunk)

        logger.info(
            f"Merged chunks: {len(text_chunks)} text + "
            f"{len(multimodal_result.get('fused_chunks', []))} multimodal = "
            f"{len(all_chunks)} total"
        )

        return all_chunks
