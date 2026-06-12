"""
retriever.py — Optimized FAISS retrieval with O(1) metadata lookup + LRU cache.
"""
import time
import json
import logging
import hashlib
from typing import Optional

import numpy as np

from backend.rag.loader import get_index, get_model, fetch_metadata

logger = logging.getLogger(__name__)

DEFAULT_TOP_K  = 3
LRU_CACHE_SIZE = 100

_cache: dict[str, str] = {}
_cache_order: list[str] = []


def _embed_query(query: str) -> np.ndarray:
    model = get_model()
    vec = model.encode(
        [query], batch_size=1, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    return vec.astype(np.float32)


def _faiss_search(query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    index = get_index()
    scores, ids = index.search(query_vec, top_k)
    return scores[0], ids[0]


def _query_hash(query: str, top_k: int) -> str:
    return hashlib.md5(f"{query}|{top_k}".encode()).hexdigest()


def _cache_get(key: str) -> Optional[list[dict]]:
    if key in _cache:
        return json.loads(_cache[key])
    return None


def _cache_set(key: str, results: list[dict]) -> None:
    if len(_cache_order) >= LRU_CACHE_SIZE:
        evict = _cache_order.pop(0)
        _cache.pop(evict, None)
    _cache[key] = json.dumps(results, ensure_ascii=False)
    _cache_order.append(key)


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Retrieve top_k documents for a query.
    Returns: [{doc_id, score, text, metadata, source}]
    """
    t0 = time.perf_counter()

    cache_key = _query_hash(query, top_k)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    query_vec = _embed_query(query)
    scores, doc_ids = _faiss_search(query_vec, top_k)

    results = []
    for score, doc_id in zip(scores, doc_ids):
        if doc_id < 0:
            continue
        entry = fetch_metadata(int(doc_id))
        results.append({
            "doc_id":   int(doc_id),
            "score":    float(score),
            "text":     entry.get("text", ""),
            "metadata": entry.get("metadata", {}),
            "source":   "faiss",
        })

    _cache_set(cache_key, results)

    logger.info(
        f"[Retriever] query='{query[:50]}' | "
        f"hits={len(results)} | total={(time.perf_counter()-t0)*1000:.1f}ms"
    )
    return results
