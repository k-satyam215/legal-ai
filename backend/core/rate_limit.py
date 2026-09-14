"""
backend/core/rate_limit.py — In-memory sliding-window rate limiter.

Protects the LLM-backed API surface (/api/v2/*) from abuse and runaway
Groq API cost. Same graceful-fallback philosophy as backend/core/cache.py:
zero external dependency, safe by default, configurable via env vars.

NOTE: state is per-process. Behind multiple worker processes/replicas this
rate-limits each worker independently rather than globally — for a
multi-instance deployment, back this with Redis the same way cache.py does.
That upgrade is intentionally left out here to avoid adding a hard Redis
dependency for a single-process/dev/HF-Spaces deployment.
"""
import os
import threading
import time
from collections import defaultdict, deque

RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def is_allowed(
    key: str,
    max_requests: int = RATE_LIMIT_MAX_REQUESTS,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> tuple[bool, int]:
    """
    Sliding-window check: drop hits older than window_seconds, then compare
    the remaining count against max_requests.

    Returns (allowed, retry_after_seconds). retry_after_seconds is 0 when
    allowed is True.
    """
    now = time.time()
    with _lock:
        q = _hits[key]
        cutoff = now - window_seconds
        while q and q[0] < cutoff:
            q.popleft()

        if len(q) >= max_requests:
            retry_after = int(window_seconds - (now - q[0])) + 1
            return False, max(retry_after, 1)

        q.append(now)
        return True, 0


def reset(key: str | None = None) -> None:
    """Clear rate-limit state. Used by tests; pass None to clear everything."""
    with _lock:
        if key is None:
            _hits.clear()
        else:
            _hits.pop(key, None)


def stats() -> dict:
    with _lock:
        return {
            "tracked_keys": len(_hits),
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "max_requests": RATE_LIMIT_MAX_REQUESTS,
        }
