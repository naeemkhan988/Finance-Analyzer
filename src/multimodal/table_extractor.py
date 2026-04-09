"""
Table Extractor
===============
Extracts and structures tables from financial documents.
Supports PDF tables, Excel sheets, and CSV data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Production table extractor for financial documents.

    Features:
    - PDF table extraction via pdfplumber
    - Excel/CSV table loading
    - Table structure normalization
    - Financial data type detection
    - Table-to-text conversion for RAG
    """

    def __init__(self):
        self._financial_headers = {
            "revenue", "sales", "income", "expenses", "profit", "loss",
            "assets", "liabilities", "equity", "cash", "debt",
            "q1", "q2", "q3", "q4", "fy", "ytd",
            "total", "net", "gross", "operating",
        }

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract all tables from a PDF file.

        Returns:
            List of table dicts with data, metadata, and text representation
        """
        tables = []

        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()

                    if page_tables:
                        for table_idx, table_data in enumerate(page_tables):
                            if not table_data or not table_data[0]:
                                continue

                            processed = self._process_table(
                                table_data, pdf_path, page_num, table_idx
                            )
                            if processed:
                                tables.append(processed)

            logger.info(
                f"Extracted {len(tables)} tables from '{Path(pdf_path).name}'"
            )

        except Exception as e:
            logger.error(f"Table extraction failed for '{pdf_path}': {e}")

        return tables

    def extract_from_excel(self, file_path: str) -> List[Dict]:
        """Extract tables from Excel files."""
        tables = []

        try:
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                if df.empty:
                    continue

                table_text = self._dataframe_to_text(df)
                is_financial = self._is_financial_table(df)

                tables.append({
                    "data": df.to_dict(orient="records"),
                    "headers": list(df.columns),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "text_representation": table_text,
                    "source": Path(file_path).name,
                    "sheet_name": sheet_name,
                    "is_financial": is_financial,
                    "type": "excel",
                })

            logger.info(
                f"Extracted {len(tables)} tables from '{Path(file_path).name}'"
            )

        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")

        return tables

    def extract_from_csv(self, file_path: str) -> List[Dict]:
        """Extract table from CSV file."""
        try:
            df = pd.read_csv(file_path)

            if df.empty:
                return []

            table_text = self._dataframe_to_text(df)
            is_financial = self._is_financial_table(df)

            return [{
                "data": df.to_dict(orient="records"),
                "headers": list(df.columns),
                "rows": len(df),
                "columns": len(df.columns),
                "text_representation": table_text,
                "source": Path(file_path).name,
                "is_financial": is_financial,
                "type": "csv",
            }]

        except Exception as e:
            logger.error(f"CSV extraction failed: {e}")
            return []

    def _process_table(
        self,
        table_data: List[List],
        source_path: str,
        page_num: int,
        table_idx: int,
    ) -> Optional[Dict]:
        """Process raw table data into structured format."""
        if len(table_data) < 2:
            return None

        # Clean headers
        headers = [str(h or "").strip() for h in table_data[0]]

        # Clean rows
        rows = []
        for row in table_data[1:]:
            cleaned = [str(c or "").strip() for c in row]
            if any(cleaned):  # Skip empty rows
                rows.append(cleaned)

        if not rows:
            return None

        # Build text representation
        text_parts = [" | ".join(headers), "-" * 40]
        for row in rows:
            text_parts.append(" | ".join(row))
        text_representation = "\n".join(text_parts)

        # Check if financial
        is_financial = any(
            h.lower() in self._financial_headers for h in headers
        )

        return {
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers),
            "text_representation": text_representation,
            "source": Path(source_path).name,
            "page": page_num,
            "table_index": table_idx,
            "is_financial": is_financial,
            "type": "pdf_table",
        }

    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert pandas DataFrame to text for RAG indexing."""
        lines = []
        headers = " | ".join(str(c) for c in df.columns)
        lines.append(headers)
        lines.append("-" * len(headers))

        for _, row in df.head(50).iterrows():  # Limit rows for text
            line = " | ".join(str(v) for v in row.values)
            lines.append(line)

        if len(df) > 50:
            lines.append(f"... ({len(df) - 50} more rows)")

        return "\n".join(lines)

    def _is_financial_table(self, df: pd.DataFrame) -> bool:
        """Detect if a table contains financial data."""
        col_names = " ".join(str(c).lower() for c in df.columns)
        return any(term in col_names for term in self._financial_headers)

    def tables_to_chunks(self, tables: List[Dict], max_chunk_size: int = 1024) -> List[Dict]:
        """
        Convert extracted tables into text chunks suitable for RAG.

        Returns list of chunk dicts with content and metadata.
        """
        chunks = []

        for table in tables:
            text = table.get("text_representation", "")
            if not text:
                continue

            if len(text) <= max_chunk_size:
                chunks.append({
                    "content": f"[TABLE]\n{text}",
                    "metadata": {
                        "source": table.get("source", "unknown"),
                        "page": table.get("page", 0),
                        "type": "table",
                        "is_financial": table.get("is_financial", False),
                        "rows": table.get("row_count", 0),
                        "columns": table.get("column_count", 0),
                    },
                })
            else:
                # Split large tables
                lines = text.split("\n")
                header = lines[0] if lines else ""
                separator = lines[1] if len(lines) > 1 else ""

                current_chunk = f"{header}\n{separator}"
                for line in lines[2:]:
                    if len(current_chunk) + len(line) + 1 > max_chunk_size:
                        chunks.append({
                            "content": f"[TABLE]\n{current_chunk}",
                            "metadata": {
                                "source": table.get("source", "unknown"),
                                "page": table.get("page", 0),
                                "type": "table_part",
                                "is_financial": table.get("is_financial", False),
                            },
                        })
                        current_chunk = f"{header}\n{separator}\n{line}"
                    else:
                        current_chunk += f"\n{line}"

                if current_chunk.strip():
                    chunks.append({
                        "content": f"[TABLE]\n{current_chunk}",
                        "metadata": {
                            "source": table.get("source", "unknown"),
                            "page": table.get("page", 0),
                            "type": "table_part",
                            "is_financial": table.get("is_financial", False),
                        },
                    })

        return chunks
