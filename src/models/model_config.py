"""
Model Configuration
===================
Centralized configuration for all AI models used in the system.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Unified model configuration."""

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LLM - Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 4096

    # LLM - Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_tokens: int = 8192

    # Generation defaults
    default_temperature: float = 0.3
    default_max_tokens: int = 2048
    retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Load configuration from environment variables."""
        return cls(
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", 384)),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            default_temperature=float(os.getenv("LLM_TEMPERATURE", 0.3)),
            default_max_tokens=int(os.getenv("LLM_MAX_TOKENS", 2048)),
        )

    def get_active_providers(self):
        """List active LLM providers based on available API keys."""
        providers = []
        if self.groq_api_key:
            providers.append({"name": "groq", "model": self.groq_model})
        if self.google_api_key:
            providers.append({"name": "gemini", "model": self.gemini_model})
        if not providers:
            providers.append({"name": "fallback", "model": "keyword_extraction"})
        return providers
