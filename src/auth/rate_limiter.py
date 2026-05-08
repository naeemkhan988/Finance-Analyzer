# Rate limiter — token bucket rate limiter with tiered limits.
"""
Rate Limiter
============
In-memory token bucket rate limiter with per-tier limits.
"""

import time
import logging
from typing import Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter with tiered access.

    Tiers:
    - free:  5/min,  20/hr,   100/day
    - pro:   20/min, 200/hr,  1000/day
    - admin: 999/min, 9999/hr, 99999/day
    """

    LIMITS = {
        "free":  {"per_minute": 5,   "per_hour": 20,   "per_day": 100},
        "pro":   {"per_minute": 20,  "per_hour": 200,  "per_day": 1000},
        "admin": {"per_minute": 999, "per_hour": 9999, "per_day": 99999},
    }

    WINDOWS = {
        "per_minute": 60,
        "per_hour": 3600,
        "per_day": 86400,
    }

    def __init__(self):
        # identifier -> window_name -> list of timestamps
        self._buckets: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

    def check(self, identifier: str, tier: str = "free") -> Tuple[bool, Dict]:
        """
        Check if a request is allowed under rate limits.

        Args:
            identifier: Unique identifier (API key hash or IP)
            tier: Access tier (free, pro, admin)

        Returns:
            Tuple of (allowed: bool, info: dict)
            If allowed: (True, {"remaining": {"per_minute": N, ...}})
            If blocked: (False, {"error": str, "retry_after": seconds})
        """
        limits = self.LIMITS.get(tier, self.LIMITS["free"])
        now = time.time()
        buckets = self._buckets[identifier]

        remaining = {}

        for window_name, limit in limits.items():
            window_seconds = self.WINDOWS[window_name]

            # Clean expired timestamps
            cutoff = now - window_seconds
            buckets[window_name] = [
                ts for ts in buckets[window_name] if ts > cutoff
            ]

            current_count = len(buckets[window_name])
            remaining[window_name] = max(0, limit - current_count)

            if current_count >= limit:
                # Calculate retry_after
                oldest = min(buckets[window_name]) if buckets[window_name] else now
                retry_after = int(oldest + window_seconds - now) + 1

                logger.warning(
                    f"Rate limit exceeded: identifier={identifier[:16]}..., "
                    f"tier={tier}, window={window_name}, "
                    f"count={current_count}/{limit}"
                )

                return False, {
                    "error": f"Rate limit exceeded ({window_name}: {limit} requests)",
                    "retry_after": max(1, retry_after),
                    "limit": limit,
                    "window": window_name,
                }

        # All windows passed — record this request
        for window_name in limits:
            buckets[window_name].append(now)

        return True, {"remaining": remaining}
