"""
Embedder — builds FAISS flat index + BM25 index from LangChain Documents.
Saves:
  backend/rag/faiss_index.index      ← custom loader uses this
  backend/rag/faiss_index_metadata.json
  backend/rag/bm25_index.pkl
"""
import os
import json
import pickle
import logging
import numpy as np

from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

FAISS_INDEX_FILE  = "backend/rag/faiss_index.index"
METADATA_JSONL    = "backend/rag/faiss_index_metadata.json"
BM25_INDEX_PATH   = "backend/rag/bm25_index.pkl"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE        = 256


def build_all_indexes(documents: list[Document]) -> None:
    """Build FAISS + BM25 indexes from LangChain Documents."""
    import faiss

    if not documents:
        logger.error("[Embedder] No documents to index.")
        return

    logger.info(f"[Embedder] Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 256

    texts = [doc.page_content for doc in documents]
    logger.info(f"[Embedder] Encoding {len(texts)} chunks...")

    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=True,
                             show_progress_bar=False, convert_to_numpy=True)
        all_vecs.append(vecs)
        if (i // BATCH_SIZE) % 10 == 0:
            logger.info(f"[Embedder] Encoded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

    matrix = np.vstack(all_vecs).astype(np.float32)
    dim    = matrix.shape[1]

    # FAISS IndexFlatIP (cosine via normalized vecs)
    index  = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, FAISS_INDEX_FILE)
    logger.info(f"[Embedder] FAISS saved: {index.ntotal} vectors → {FAISS_INDEX_FILE}")

    # Metadata JSONL
    with open(METADATA_JSONL, "w", encoding="utf-8") as f:
        for doc in documents:
            line = json.dumps({"text": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False)
            f.write(line + "\n")
    logger.info(f"[Embedder] Metadata saved → {METADATA_JSONL}")

    # BM25
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": bm25, "documents": documents}, f)
        logger.info(f"[Embedder] BM25 saved → {BM25_INDEX_PATH}")
    except ImportError:
        logger.warning("[Embedder] rank_bm25 not installed — skipping BM25.")

    logger.info("[Embedder] ✅ All indexes built.")
