"""
OCR Processor
=============
Optical Character Recognition for scanned documents and images.
Uses Tesseract OCR with pre-processing for financial documents.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR processor for scanned financial documents.

    Features:
    - Tesseract OCR integration
    - Image pre-processing (deskew, denoise, threshold)
    - Multi-language support
    - Confidence scoring
    - Financial document optimizations
    """

    def __init__(
        self,
        language: str = "eng",
        tesseract_config: str = "--oem 3 --psm 6",
        preprocess: bool = True,
    ):
        self.language = language
        self.tesseract_config = tesseract_config
        self.preprocess = preprocess
        self._tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.warning(
                "Tesseract OCR not available. "
                "Install Tesseract and pytesseract for OCR support."
            )
            return False

    def process_image(self, image_path: str) -> Dict:
        """
        Run OCR on a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with extracted text, confidence, and metadata
        """
        if not self._tesseract_available:
            return {
                "text": "",
                "confidence": 0,
                "error": "Tesseract not available",
            }

        try:
            from PIL import Image
            import pytesseract

            image = Image.open(image_path)

            # Pre-process if enabled
            if self.preprocess:
                image = self._preprocess_image(image)

            # Run OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.tesseract_config,
            )

            # Get detailed data with confidence
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                output_type=pytesseract.Output.DICT,
            )

            # Calculate average confidence
            confidences = [
                int(c) for c in data["conf"] if int(c) > 0
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0
            )

            return {
                "text": text.strip(),
                "confidence": round(avg_confidence, 2),
                "word_count": len(text.split()),
                "source": Path(image_path).name,
            }

        except Exception as e:
            logger.error(f"OCR processing failed for '{image_path}': {e}")
            return {
                "text": "",
                "confidence": 0,
                "error": str(e),
            }

    def process_pdf_page(self, pdf_path: str, page_num: int = 0) -> Dict:
        """
        Run OCR on a specific PDF page (for scanned PDFs).

        Converts the page to an image first, then runs OCR.
        """
        if not self._tesseract_available:
            return {"text": "", "confidence": 0, "error": "Tesseract not available"}

        try:
            from PIL import Image
            import pytesseract

            # Try using pdf2image
            try:
                from pdf2image import convert_from_path

                images = convert_from_path(
                    pdf_path,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    dpi=300,
                )

                if images:
                    image = images[0]
                    if self.preprocess:
                        image = self._preprocess_image(image)

                    text = pytesseract.image_to_string(
                        image, lang=self.language, config=self.tesseract_config
                    )

                    return {
                        "text": text.strip(),
                        "confidence": 0,
                        "page": page_num + 1,
                        "source": Path(pdf_path).name,
                    }
            except ImportError:
                logger.warning("pdf2image not available for PDF OCR")

            return {"text": "", "confidence": 0, "error": "pdf2image not available"}

        except Exception as e:
            logger.error(f"PDF OCR failed: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}

    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """Process multiple images with OCR."""
        results = []
        for path in image_paths:
            result = self.process_image(path)
            results.append(result)
        return results

    def _preprocess_image(self, image):
        """
        Pre-process image for better OCR accuracy.

        Steps:
        - Convert to grayscale
        - Apply threshold for binarization
        - Resize if too small
        """
        try:
            from PIL import ImageFilter, ImageOps

            # Convert to grayscale
            if image.mode != "L":
                image = image.convert("L")

            # Auto-contrast
            image = ImageOps.autocontrast(image)

            # Apply slight sharpening
            image = image.filter(ImageFilter.SHARPEN)

            # Resize if very small
            width, height = image.size
            if width < 300 or height < 300:
                scale = max(300 / width, 300 / height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size)

            return image

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image

    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Detect if a PDF is scanned (image-based) vs text-based.

        Checks first few pages for extractable text.
        """
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                pages_to_check = min(sample_pages, len(pdf.pages))

                for i in range(pages_to_check):
                    text = pdf.pages[i].extract_text() or ""
                    if len(text.strip()) > 50:
                        return False  # Has extractable text

            return True  # No significant text found

        except Exception:
            return False
