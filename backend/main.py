"""
FastAPI Main Application
========================
Entry point for the Multimodal RAG Finance Analyzer API.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.upload import router as upload_router
from backend.routes.query import router as query_router
from backend.routes.health import router as health_router
from backend.core.config import settings
from backend.core.dependencies import get_rag_pipeline
from src.utils.logger import setup_logging

# Setup logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Multimodal RAG Finance Analyzer API...")
    logger.info(f"Environment: {os.getenv('FLASK_ENV', 'development')}")

    # Initialize RAG pipeline
    try:
        pipeline = get_rag_pipeline()
        app.state.rag_pipeline = pipeline
        logger.info("✅ RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ RAG Pipeline initialization failed: {e}")
        app.state.rag_pipeline = None

    yield

    # Shutdown
    logger.info("🛑 Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG Finance Analyzer",
    description="Production-ready RAG system for intelligent financial document analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(upload_router, prefix="/api", tags=["Documents"])
app.include_router(query_router, prefix="/api", tags=["Query"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Multimodal RAG Finance Analyzer",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


def main():
    """Run the server directly."""
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
