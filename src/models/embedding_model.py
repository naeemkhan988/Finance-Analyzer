"""
Embedding Model Configuration
==============================
Configuration and management for embedding models.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 64
    normalize: bool = True
    device: str = "cpu"
    max_seq_length: int = 256

    # Available pre-configured models
    AVAILABLE_MODELS: Dict = field(default_factory=lambda: {
        "all-MiniLM-L6-v2": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "max_seq_length": 256,
            "description": "Fast, lightweight. Good for general use.",
        },
        "all-mpnet-base-v2": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "max_seq_length": 384,
            "description": "Higher quality, slower. Best for accuracy.",
        },
        "paraphrase-MiniLM-L3-v2": {
            "name": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "dimension": 384,
            "max_seq_length": 128,
            "description": "Fastest, smallest. Good for quick prototyping.",
        },
        "bge-small-en-v1.5": {
            "name": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "max_seq_length": 512,
            "description": "Strong performance for retrieval tasks.",
        },
    })

    @classmethod
    def from_preset(cls, preset_name: str) -> "EmbeddingModelConfig":
        """Create config from a preset model name."""
        config = cls()
        presets = config.AVAILABLE_MODELS

        if preset_name in presets:
            preset = presets[preset_name]
            return cls(
                model_name=preset["name"],
                dimension=preset["dimension"],
                max_seq_length=preset["max_seq_length"],
            )

        logger.warning(f"Unknown preset '{preset_name}', using default.")
        return cls()

    def get_model_info(self) -> Dict:
        """Get current model information."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "device": self.device,
            "max_seq_length": self.max_seq_length,
        }
