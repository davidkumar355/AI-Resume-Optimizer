import time
from collections import defaultdict
from typing import Dict, List
from fastapi import Request, HTTPException, status

from config import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


class InMemoryRateLimiter:
    def __init__(self, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.client_history: Dict[str, List[float]] = defaultdict(list)

    def is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds

        # Filter timestamps outside the current sliding window
        self.client_history[client_ip] = [
            t for t in self.client_history[client_ip] if t > window_start
        ]

        if len(self.client_history[client_ip]) >= self.max_requests:
            return True

        self.client_history[client_ip].append(now)
        return False


# Global instance
rate_limiter = InMemoryRateLimiter()


async def check_rate_limit(request: Request):
    """FastAPI dependency to enforce rate limits per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS} runs per {RATE_LIMIT_WINDOW_SECONDS} seconds.",
        )
