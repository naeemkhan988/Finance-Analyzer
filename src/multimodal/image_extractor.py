"""
Image Extractor
===============
Extracts images from PDF documents for multimodal analysis.
Supports chart detection and financial figure extraction.
"""

import io
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Extracts and processes images from financial documents.

    Features:
    - PDF image extraction using pdfplumber/PyMuPDF
    - Chart and graph detection
    - Image metadata enrichment
    - Thumbnail generation
    - Financial figure classification
    """

    def __init__(
        self,
        output_dir: str = "./data/processed/images",
        min_width: int = 100,
        min_height: int = 100,
        supported_formats: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_width = min_width
        self.min_height = min_height
        self.supported_formats = supported_formats or ["png", "jpeg", "jpg"]

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract all images from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of image metadata dicts with paths and info
        """
        images = []
        pdf_name = Path(pdf_path).stem

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_images = page.images
                    for img_idx, img in enumerate(page_images):
                        # Filter by minimum size
                        width = img.get("width", 0)
                        height = img.get("height", 0)

                        if width < self.min_width or height < self.min_height:
                            continue

                        image_info = {
                            "source_pdf": pdf_name,
                            "page": page_num,
                            "image_index": img_idx,
                            "width": width,
                            "height": height,
                            "x0": img.get("x0", 0),
                            "y0": img.get("top", 0),
                            "x1": img.get("x1", 0),
                            "y1": img.get("bottom", 0),
                            "type": self._classify_image(width, height),
                        }

                        images.append(image_info)

            logger.info(
                f"Extracted {len(images)} images from '{pdf_name}'"
            )

        except ImportError:
            logger.warning("pdfplumber not available for image extraction")
        except Exception as e:
            logger.error(f"Image extraction failed for '{pdf_path}': {e}")

        return images

    def extract_from_pdf_advanced(self, pdf_path: str) -> List[Dict]:
        """
        Advanced image extraction using PyMuPDF (fitz) for higher quality.

        Returns:
            List of dicts with image data, metadata, and saved paths
        """
        images = []
        pdf_name = Path(pdf_path).stem

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)

                        if width < self.min_width or height < self.min_height:
                            continue

                        # Save image
                        img_filename = f"{pdf_name}_p{page_num + 1}_img{img_idx}.{image_ext}"
                        img_path = self.output_dir / img_filename

                        with open(img_path, "wb") as f:
                            f.write(image_bytes)

                        images.append({
                            "source_pdf": pdf_name,
                            "page": page_num + 1,
                            "image_index": img_idx,
                            "width": width,
                            "height": height,
                            "format": image_ext,
                            "file_path": str(img_path),
                            "size_bytes": len(image_bytes),
                            "type": self._classify_image(width, height),
                        })

            doc.close()
            logger.info(
                f"Advanced extraction: {len(images)} images from '{pdf_name}'"
            )

        except ImportError:
            logger.warning("PyMuPDF not available. Using basic extraction.")
            return self.extract_from_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Advanced image extraction failed: {e}")

        return images

    def _classify_image(self, width: int, height: int) -> str:
        """
        Classify image type based on dimensions.

        Financial documents typically contain:
        - Charts/graphs (wider than tall, medium-large)
        - Logos (small, roughly square)
        - Signatures (small, wider than tall)
        - Full-page figures (large)
        """
        aspect_ratio = width / max(height, 1)
        area = width * height

        if area < 5000:
            return "icon_or_logo"
        elif area < 15000 and aspect_ratio > 2:
            return "signature_or_banner"
        elif aspect_ratio > 1.2 and area > 20000:
            return "chart_or_graph"
        elif area > 100000:
            return "full_page_figure"
        else:
            return "illustration"

    def get_image_stats(self, images: List[Dict]) -> Dict:
        """Get summary statistics about extracted images."""
        if not images:
            return {"total_images": 0}

        types = {}
        for img in images:
            img_type = img.get("type", "unknown")
            types[img_type] = types.get(img_type, 0) + 1

        return {
            "total_images": len(images),
            "by_type": types,
            "pages_with_images": len(set(img["page"] for img in images)),
            "chart_count": types.get("chart_or_graph", 0),
        }
