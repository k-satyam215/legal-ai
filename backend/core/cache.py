import hashlib, time, logging, threading
from typing import Optional
logger = logging.getLogger(__name__)
_cache: dict[str, dict] = {}
_lock = threading.Lock()
DEFAULT_TTL = 3600
MAX_CACHE_SIZE = 500

def _key(query: str, case_type: str = "") -> str:
    return hashlib.sha256(f"{query.lower().strip()}|{case_type}".encode()).hexdigest()[:16]

def cache_get(query: str, case_type: str = "") -> Optional[dict]:
    k = _key(query, case_type)
    with _lock:
        e = _cache.get(k)
        if e is None: return None
        if time.time() > e["expires_at"]:
            del _cache[k]; return None
        return e["data"]

def cache_set(query: str, data: dict, case_type: str = "", ttl: int = DEFAULT_TTL):
    k = _key(query, case_type)
    with _lock:
        if len(_cache) >= MAX_CACHE_SIZE:
            del _cache[min(_cache, key=lambda x: _cache[x]["created_at"])]
        _cache[k] = {"data": data, "created_at": time.time(), "expires_at": time.time()+ttl}

def cache_stats() -> dict:
    with _lock:
        now = time.time()
        return {"total_keys": len(_cache), "active_keys": sum(1 for e in _cache.values() if now < e["expires_at"])}

def cache_clear():
    with _lock: _cache.clear()
