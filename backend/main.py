# FastAPI main application — entry point with auth middleware, rate limiting, and startup checks.
"""
FastAPI Main Application
========================
Entry point for the Multimodal RAG Finance Analyzer API.
Includes API key authentication, rate limiting, and startup validation.
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

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes.upload import router as upload_router
from backend.routes.query import router as query_router
from backend.routes.health import router as health_router
from backend.core.config import settings
from backend.core.dependencies import get_rag_pipeline
from src.utils.logger import setup_logging

# Setup logging
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Auth components (initialized lazily)
_api_key_manager = None
_rate_limiter = None


def _get_api_key_manager():
    global _api_key_manager
    if _api_key_manager is None:
        try:
            from src.auth.api_key_manager import APIKeyManager
            _api_key_manager = APIKeyManager()
        except Exception as e:
            logger.warning(f"APIKeyManager init failed: {e}")
    return _api_key_manager


def _get_rate_limiter():
    global _rate_limiter
    if _rate_limiter is None:
        try:
            from src.auth.rate_limiter import RateLimiter
            _rate_limiter = RateLimiter()
        except Exception as e:
            logger.warning(f"RateLimiter init failed: {e}")
    return _rate_limiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("Starting Multimodal RAG Finance Analyzer API...")
    logger.info(f"Environment: {os.getenv('FLASK_ENV', 'development')}")

    # Run startup checks
    try:
        from src.utils.startup_check import run_startup_checks
        checks_ok = run_startup_checks()
        if not checks_ok:
            logger.critical(
                "Startup checks failed. Continuing with degraded functionality."
            )
    except Exception as e:
        logger.error(f"Startup checks error: {e}")

    # Initialize auth components
    _get_api_key_manager()
    _get_rate_limiter()

    # Initialize RAG pipeline
    try:
        pipeline = get_rag_pipeline()
        app.state.rag_pipeline = pipeline
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"RAG Pipeline initialization failed: {e}")
        app.state.rag_pipeline = None

    yield

    # Shutdown
    logger.info("Shutting down API server...")


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


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    API key authentication and rate limiting middleware.
    Applies to all /api/ routes except /api/health.
    """
    path = request.url.path

    # Only protect /api/ routes (except health)
    if path.startswith("/api/") and path != "/api/health":
        key_manager = _get_api_key_manager()
        limiter = _get_rate_limiter()

        if key_manager is not None:
            api_key = request.headers.get("X-API-Key", "")

            # Validate key
            key_info = key_manager.validate(api_key)
            if not key_info["valid"]:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Invalid or missing API key",
                        "detail": "Provide a valid API key via X-API-Key header",
                    },
                )

            # Check rate limit
            if limiter is not None:
                identifier = api_key[:16] if api_key else (
                    request.client.host if request.client else "unknown"
                )
                allowed, info = limiter.check(identifier, key_info["tier"])

                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": info.get("error", "Rate limit exceeded"),
                            "retry_after": info.get("retry_after", 60),
                        },
                        headers={
                            "Retry-After": str(info.get("retry_after", 60)),
                        },
                    )

                # Add rate limit headers to response
                response = await call_next(request)
                remaining = info.get("remaining", {})
                response.headers["X-RateLimit-Remaining"] = str(
                    remaining.get("per_minute", 0)
                )
                return response

    response = await call_next(request)
    return response


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
