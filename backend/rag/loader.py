"""
loader.py — One-time initialization: FAISS index + byte-offset metadata.
Call initialize() once at FastAPI startup. All other modules use get_index() / get_model().
"""
import os
import time
import json
import logging
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "backend/rag/faiss_index.index")
METADATA_JSONL   = os.getenv("METADATA_JSONL",   "backend/rag/faiss_index_metadata.json")
OFFSETS_FILE     = os.getenv("OFFSETS_FILE",      "backend/rag/faiss_index_offsets.npy")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",   "sentence-transformers/all-MiniLM-L6-v2")

_faiss_index:  Optional[faiss.Index]         = None
_st_model:     Optional[SentenceTransformer] = None
_meta_offsets: Optional[np.ndarray]          = None
_meta_file:    Optional[object]              = None


def _build_offset_index(jsonl_path: str, offsets_path: str) -> np.ndarray:
    logger.info(f"[Loader] Building byte-offset index from {jsonl_path}...")
    t0 = time.time()
    offsets = []
    with open(jsonl_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    arr = np.array(offsets, dtype=np.uint64)
    np.save(offsets_path, arr)
    logger.info(f"[Loader] Offset index built: {len(arr):,} entries | took {time.time()-t0:.1f}s")
    return arr


def _load_offset_index(jsonl_path: str, offsets_path: str) -> np.ndarray:
    if os.path.exists(offsets_path):
        if os.path.getmtime(offsets_path) >= os.path.getmtime(jsonl_path):
            arr = np.load(offsets_path)
            logger.info(f"[Loader] Loaded offset index: {len(arr):,} entries")
            return arr
        logger.info("[Loader] Offset index stale — rebuilding.")
    return _build_offset_index(jsonl_path, offsets_path)


def fetch_metadata(doc_id: int) -> dict:
    global _meta_offsets, _meta_file
    if _meta_offsets is None or _meta_file is None:
        raise RuntimeError("Call initialize() first.")
    if doc_id < 0 or doc_id >= len(_meta_offsets):
        return {"text": "", "metadata": {}}
    _meta_file.seek(int(_meta_offsets[doc_id]))
    return json.loads(_meta_file.readline())


def initialize(
    faiss_path:   str = FAISS_INDEX_FILE,
    jsonl_path:   str = METADATA_JSONL,
    offsets_path: str = OFFSETS_FILE,
    model_name:   str = EMBEDDING_MODEL,
) -> None:
    global _faiss_index, _st_model, _meta_offsets, _meta_file
    if _faiss_index is not None:
        return

    t0 = time.time()
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    logger.info(f"[Loader] Loading FAISS index...")
    _faiss_index = faiss.read_index(faiss_path)
    logger.info(f"[Loader] FAISS: {_faiss_index.ntotal:,} vectors")

    logger.info(f"[Loader] Loading SentenceTransformer: {model_name}")
    _st_model = SentenceTransformer(model_name)
    _st_model.max_seq_length = 256

    _meta_offsets = _load_offset_index(jsonl_path, offsets_path)
    _meta_file = open(jsonl_path, "rb")
    logger.info(f"[Loader] ✅ Ready in {time.time()-t0:.2f}s")


def get_index() -> faiss.Index:
    if _faiss_index is None:
        raise RuntimeError("Call initialize() first.")
    return _faiss_index


def get_model() -> SentenceTransformer:
    if _st_model is None:
        raise RuntimeError("Call initialize() first.")
    return _st_model
