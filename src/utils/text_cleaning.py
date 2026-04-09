"""
Text Cleaning Utilities
=======================
Text preprocessing and cleaning for financial documents.
"""

import re
import unicodedata
from typing import List, Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize text from financial documents.

    Steps:
    - Normalize unicode characters
    - Remove control characters
    - Normalize whitespace
    - Fix common OCR artifacts
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Fix common OCR artifacts
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.replace("ﬀ", "ff")

    # Normalize dashes
    text = re.sub(r"[–—]", "-", text)

    # Normalize quotes
    text = re.sub(r"["""]", '"', text)
    text = re.sub(r"[''']", "'", text)

    # Normalize whitespace (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_headers_footers(text: str, patterns: Optional[List[str]] = None) -> str:
    """
    Remove common headers and footers from document text.

    Common patterns in financial docs:
    - Page numbers
    - Confidential notices
    - Company names repeated on every page
    """
    if not text:
        return ""

    # Default patterns
    default_patterns = [
        r"^Page\s+\d+\s*(of\s+\d+)?$",
        r"^\d+\s*$",  # Standalone page numbers
        r"(?i)^confidential\s*$",
        r"(?i)^draft\s*$",
        r"(?i)^for internal use only\s*$",
    ]

    all_patterns = default_patterns + (patterns or [])

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not any(re.match(p, stripped) for p in all_patterns):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_financial_numbers(text: str) -> str:
    """
    Normalize financial number formatting for consistency.

    Examples:
    - "$1,234.56" stays as-is
    - "$ 1,234.56" -> "$1,234.56"
    - "1234.56 USD" -> "$1,234.56"
    """
    # Fix spacing after currency symbols
    text = re.sub(r"\$\s+", "$", text)
    text = re.sub(r"€\s+", "€", text)
    text = re.sub(r"£\s+", "£", text)

    return text


def extract_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []

    # Simple sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text at word boundary."""
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.5:
        truncated = truncated[:last_space]

    return truncated + suffix


def count_tokens_estimate(text: str) -> int:
    """
    Rough token count estimate (1 token ≈ 4 characters for English).
    For exact counts, use tiktoken.
    """
    return len(text) // 4
