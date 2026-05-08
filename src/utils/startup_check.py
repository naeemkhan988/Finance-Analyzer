# Startup checks — pre-flight validation for application dependencies and configuration.
"""
Startup Checks
==============
Pre-flight validation ensuring all required dependencies,
environment variables, and directories are properly configured.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_startup_checks() -> bool:
    """
    Run all startup validation checks.

    Returns:
        True if no critical errors found, False if any critical check failed.
    """
    errors = []
    warnings = []

    # 1. Check SECRET_KEY
    secret = os.getenv("SECRET_KEY", "")
    if not secret or secret == "dev-secret-key-change-in-production":
        errors.append(
            "SECRET_KEY is not set or using default value. "
            "Set a strong SECRET_KEY in your .env file for production."
        )

    # 2. Check LLM API keys
    groq_key = os.getenv("GROQ_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if not groq_key and not google_key:
        warnings.append(
            "No LLM API keys configured (GROQ_API_KEY or GOOGLE_API_KEY). "
            "The system will run in fallback mode with keyword-based extraction only. "
            "Set at least one API key in .env for AI-powered analysis."
        )

    # 3. Check required directories
    required_dirs = [
        "./data",
        "./data/raw",
        "./data/embeddings",
        "./logs",
    ]
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                errors.append(f"Failed to create directory '{dir_path}': {e}")

    # 4. Check sentence-transformers
    try:
        import sentence_transformers  # noqa: F401
        logger.info("sentence-transformers: available")
    except ImportError:
        errors.append(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        )

    # 5. Check FAISS
    try:
        import faiss  # noqa: F401
        logger.info("faiss-cpu: available")
    except ImportError:
        warnings.append(
            "faiss-cpu is not installed. Using numpy fallback for vector search. "
            "Install with: pip install faiss-cpu"
        )

    # 6. Check Tesseract OCR
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("tesseract: available")
    except Exception:
        warnings.append(
            "Tesseract OCR is not installed or not in PATH. "
            "Multimodal features (scanned PDF/Chart OCR) will be disabled. "
            "Install Tesseract and add it to your PATH to enable full multimodal support."
        )

    # Log results
    for w in warnings:
        logger.warning(f"STARTUP WARNING: {w}")

    for e in errors:
        logger.error(f"STARTUP ERROR: {e}")

    if errors:
        logger.error(
            f"Startup checks: {len(errors)} error(s), {len(warnings)} warning(s). "
            f"System may not function correctly."
        )
        return False

    logger.info(
        f"Startup checks passed: 0 errors, {len(warnings)} warning(s)"
    )
    return True
