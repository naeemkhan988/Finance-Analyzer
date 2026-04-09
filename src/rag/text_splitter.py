"""
Text Splitter
=============
Advanced text splitting strategies for financial documents.
Supports recursive, sentence-based, and semantic splitting.
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for text splitting."""
    chunk_size: int = 512
    chunk_overlap: int = 128
    separators: Optional[List[str]] = None
    keep_separator: bool = True
    strip_whitespace: bool = True

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", ", ", " ", ""]


class TextSplitter:
    """
    Production text splitter with multiple splitting strategies.

    Strategies:
    - Recursive character splitting (default)
    - Sentence-based splitting
    - Fixed-size splitting with overlap
    - Financial-aware splitting (respects table boundaries)
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()

    def split_text(self, text: str) -> List[str]:
        """
        Split text using recursive character strategy.

        Tries each separator in order, splitting on the first one
        that produces chunks within the target size.
        """
        return self._recursive_split(text, self.config.separators)

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text on sentence boundaries."""
        sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)'
        )
        sentences = sentence_pattern.split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            test_chunk = current_chunk + (" " if current_chunk else "") + sentence

            if len(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    def split_financial_text(self, text: str) -> List[str]:
        """
        Financial-aware splitting that respects table and section boundaries.

        Preserves:
        - Complete tables (delimited by [TABLES] markers)
        - Section headers and their content
        - Numerical data integrity
        """
        # Separate table sections from text
        table_marker = "[TABLES]"
        parts = text.split(table_marker)

        all_chunks = []

        for i, part in enumerate(parts):
            if i > 0:
                # This is a table section - try to keep it intact
                if len(part) <= self.config.chunk_size * 2:
                    all_chunks.append(f"{table_marker}\n{part.strip()}")
                else:
                    # Table is too large, split by rows
                    table_chunks = self._split_table(part)
                    all_chunks.extend(table_chunks)
            else:
                # Regular text section
                text_chunks = self.split_text(part)
                all_chunks.extend(text_chunks)

        return [c for c in all_chunks if c.strip()]

    def _split_table(self, table_text: str) -> List[str]:
        """Split a large table while preserving header row."""
        lines = table_text.strip().split("\n")
        if len(lines) <= 2:
            return [table_text.strip()] if table_text.strip() else []

        header = lines[0]
        separator = lines[1] if lines[1].startswith("-") else ""

        chunks = []
        current_rows = [header]
        if separator:
            current_rows.append(separator)

        for line in lines[2:] if separator else lines[1:]:
            test_chunk = "\n".join(current_rows + [line])
            if len(test_chunk) <= self.config.chunk_size:
                current_rows.append(line)
            else:
                chunks.append("[TABLE]\n" + "\n".join(current_rows))
                current_rows = [header]
                if separator:
                    current_rows.append(separator)
                current_rows.append(line)

        if len(current_rows) > (2 if separator else 1):
            chunks.append("[TABLE]\n" + "\n".join(current_rows))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using multiple separators."""
        if len(text) <= self.config.chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0] if separators else ""
        chunks = []
        current_chunk = ""

        parts = text.split(sep) if sep else list(text)

        for part in parts:
            connector = sep if (current_chunk and self.config.keep_separator) else ""
            test_chunk = current_chunk + connector + part

            if len(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip() if self.config.strip_whitespace else current_chunk)

                    # Overlap handling
                    if self.config.chunk_overlap > 0:
                        overlap = current_chunk[-self.config.chunk_overlap:]
                        current_chunk = overlap + connector + part
                    else:
                        current_chunk = part
                else:
                    # Part too large, use next separator
                    if len(separators) > 1:
                        sub_chunks = self._recursive_split(part, separators[1:])
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        for j in range(0, len(part), self.config.chunk_size):
                            chunk = part[j: j + self.config.chunk_size]
                            if chunk.strip():
                                chunks.append(chunk)
                        current_chunk = ""

        if current_chunk.strip():
            chunks.append(current_chunk.strip() if self.config.strip_whitespace else current_chunk)

        return chunks

    def estimate_chunks(self, text: str) -> int:
        """Estimate the number of chunks without actually splitting."""
        if not text:
            return 0
        effective_size = self.config.chunk_size - self.config.chunk_overlap
        return max(1, len(text) // effective_size)
