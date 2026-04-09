"""
Logging Middleware
=================
Request/response logging for API monitoring.
"""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests and responses.

    Logs:
    - Request method, path, and client IP
    - Response status code
    - Request processing time
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} "
            f"- {process_time:.3f}s "
            f"- {request.client.host if request.client else 'unknown'}"
        )

        # Add processing time header
        response.headers["X-Process-Time"] = str(round(process_time, 3))

        return response
