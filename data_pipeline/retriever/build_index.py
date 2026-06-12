"""CLI: build FAISS + BM25 indexes from PDFs in backend/data/"""
import os
import sys
import logging
import fitz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_pipeline.cleaner.clean_text import clean_pipeline
from data_pipeline.chunker.chunking import chunk_all_documents
from data_pipeline.embeddings.embedder import build_all_indexes
from backend.core.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_pdfs_from_disk() -> list[dict]:
    docs = []
    data_path = str(DATA_DIR)
    if not os.path.exists(data_path):
        logger.warning(f"DATA_DIR not found: {data_path}")
        return docs
    for root, _, files in os.walk(data_path):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fname)
            try:
                pdf  = fitz.open(path)
                text = "".join(page.get_text("text") + "\n" for page in pdf)
                pdf.close()
                if len(text.strip()) < 100:
                    continue
                year = os.path.basename(root) if os.path.basename(root).isdigit() else "Unknown"
                docs.append({
                    "text": text,
                    "law_name": fname.replace("_", " ").replace(".PDF", "").replace(".pdf", ""),
                    "source": path,
                    "date": year,
                    "court": "Supreme Court of India",
                    "category": "general",
                    "section": "Full Judgment",
                })
            except Exception as e:
                logger.error(f"Failed to read {path}: {e}")
    logger.info(f"[BuildIndex] Loaded {len(docs)} PDFs")
    return docs


def build_index_from_pdfs():
    raw_docs = load_pdfs_from_disk()
    if not raw_docs:
        logger.error("No documents found.")
        return
    cleaned  = clean_pipeline(raw_docs)
    lc_docs  = chunk_all_documents(cleaned)
    build_all_indexes(lc_docs)
    logger.info("[BuildIndex] ✅ Done.")


if __name__ == "__main__":
    build_index_from_pdfs()
