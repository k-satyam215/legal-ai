"""
backend/core/cache.py — Redis-backed TTL cache with graceful in-memory fallback.

Same public interface as before:
  cache_get(), cache_set(), cache_stats(), cache_clear()

So router_v2.py and routes.py need ZERO changes.

Behaviour:
  - If Redis is running  → data stored in Redis (persists across restarts)
  - If Redis is down     → silently falls back to in-memory dict (app never crashes)
"""
import hashlib
import json
import logging
import os
import threading
import time
from typing import Optional

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_TTL    = 3600
MAX_CACHE_SIZE = 500
CACHE_PREFIX   = "legal_advisor:"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# ── In-memory fallback (original implementation) ──────────────────────────
_cache: dict[str, dict] = {}
_lock  = threading.Lock()

# ── Redis client (lazy, so startup never fails if Redis is absent) ─────────
_redis_client: Optional[object] = None
_redis_ok = True   # flipped to False on first error, avoids retry spam


def _client() -> Optional[object]:
    """Return a live Redis client, or None if Redis is unavailable."""
    global _redis_client, _redis_ok
    if not _REDIS_AVAILABLE or not _redis_ok:
        return None
    if _redis_client is None:
        try:
            r = _redis_lib.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            r.ping()
            _redis_client = r
            logger.info(f"[Cache] Redis connected at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as exc:
            logger.warning(f"[Cache] Redis unavailable ({exc}) — using in-memory fallback")
            _redis_ok = False
    return _redis_client


def _key(query: str, case_type: str = "") -> str:
    return hashlib.sha256(f"{query.lower().strip()}|{case_type}".encode()).hexdigest()[:16]


# ── In-memory helpers ──────────────────────────────────────────────────────
def _mem_get(k: str) -> Optional[dict]:
    with _lock:
        e = _cache.get(k)
        if e is None:
            return None
        if time.time() > e["expires_at"]:
            del _cache[k]
            return None
        return e["data"]


def _mem_set(k: str, data: dict, ttl: int) -> None:
    with _lock:
        if len(_cache) >= MAX_CACHE_SIZE:
            del _cache[min(_cache, key=lambda x: _cache[x]["created_at"])]
        _cache[k] = {"data": data, "created_at": time.time(), "expires_at": time.time() + ttl}


# ── Public interface (identical signatures to original) ────────────────────
def cache_get(query: str, case_type: str = "") -> Optional[dict]:
    k = _key(query, case_type)
    r = _client()
    if r is not None:
        try:
            raw = r.get(CACHE_PREFIX + k)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning(f"[Cache] Redis GET failed ({exc}), using in-memory")
    return _mem_get(k)


def cache_set(query: str, data: dict, case_type: str = "", ttl: int = DEFAULT_TTL) -> None:
    k = _key(query, case_type)
    r = _client()
    if r is not None:
        try:
            r.setex(CACHE_PREFIX + k, ttl, json.dumps(data))
            return
        except Exception as exc:
            logger.warning(f"[Cache] Redis SET failed ({exc}), using in-memory")
    _mem_set(k, data, ttl)


def cache_stats() -> dict:
    r = _client()
    if r is not None:
        try:
            keys = r.keys(CACHE_PREFIX + "*")
            return {"backend": "redis", "host": REDIS_HOST, "port": REDIS_PORT, "active_keys": len(keys)}
        except Exception as exc:
            logger.warning(f"[Cache] Redis STATS failed ({exc})")
    with _lock:
        now = time.time()
        return {
            "backend": "in-memory (redis unavailable)",
            "total_keys": len(_cache),
            "active_keys": sum(1 for e in _cache.values() if now < e["expires_at"]),
        }


def cache_clear() -> None:
    r = _client()
    if r is not None:
        try:
            keys = r.keys(CACHE_PREFIX + "*")
            if keys:
                r.delete(*keys)
        except Exception as exc:
            logger.warning(f"[Cache] Redis CLEAR failed ({exc})")
    with _lock:
        _cache.clear()
