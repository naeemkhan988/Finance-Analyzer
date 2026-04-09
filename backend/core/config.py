"""
Backend Configuration
=====================
Application settings loaded from environment variables.
"""

import os
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings."""

    # Server
    HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("APP_PORT", 8000))
    DEBUG: bool = os.getenv("FLASK_DEBUG", "0") == "1"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Upload
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "./data/raw")
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 50))

    # RAG Pipeline
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 128))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 384))
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/embeddings")
    FAISS_INDEX_NAME: str = os.getenv("FAISS_INDEX_NAME", "financial_docs")

    class Config:
        env_file = ".env"


settings = Settings()
