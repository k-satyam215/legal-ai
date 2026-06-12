"""
Master data pipeline orchestrator.
Modes:
  --mode scrape  → scrape + clean + chunk + index
  --mode pdf     → use existing PDFs in backend/data/ (offline)
  --mode cache   → re-index from saved raw JSON

Usage:
  python data_pipeline/pipeline.py --mode pdf
  python data_pipeline/pipeline.py --mode scrape
"""
import sys
import os
import json
import logging
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.cleaner.clean_text import clean_pipeline
from data_pipeline.chunker.chunking import chunk_all_documents
from data_pipeline.embeddings.embedder import build_all_indexes
from data_pipeline.retriever.build_index import load_pdfs_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_CACHE = "data_pipeline/raw_scraped.json"


def pipeline_from_pdfs():
    logger.info("=== MODE: PDF ===")
    raw_docs = load_pdfs_from_disk()
    if not raw_docs:
        logger.error("No PDFs found in backend/data/. Aborting.")
        return
    _run_pipeline(raw_docs)


def pipeline_from_scrape():
    logger.info("=== MODE: SCRAPE ===")
    from data_pipeline.scraper.indiankanoon_scraper import scrape_all_categories
    from data_pipeline.scraper.indiancode_scraper import scrape_target_acts

    case_laws    = scrape_all_categories(max_pages_per_cat=2)
    act_sections = scrape_target_acts()
    raw_docs     = case_laws + act_sections
    logger.info(f"Total raw documents: {len(raw_docs)}")

    with open(RAW_DATA_CACHE, "w") as f:
        json.dump(raw_docs, f, ensure_ascii=False, indent=2)
    logger.info(f"Cached → {RAW_DATA_CACHE}")
    _run_pipeline(raw_docs)


def pipeline_from_cache():
    logger.info("=== MODE: CACHE ===")
    if not os.path.exists(RAW_DATA_CACHE):
        logger.error(f"Cache not found: {RAW_DATA_CACHE}. Run --mode scrape first.")
        return
    with open(RAW_DATA_CACHE) as f:
        raw_docs = json.load(f)
    _run_pipeline(raw_docs)


def _run_pipeline(raw_docs: list[dict]):
    logger.info(f"[Pipeline] Cleaning {len(raw_docs)} docs...")
    cleaned  = clean_pipeline(raw_docs)
    logger.info(f"[Pipeline] After cleaning: {len(cleaned)} docs")

    logger.info("[Pipeline] Chunking...")
    lc_docs  = chunk_all_documents(cleaned)
    logger.info(f"[Pipeline] Total chunks: {len(lc_docs)}")

    if not lc_docs:
        logger.error("[Pipeline] No chunks. Aborting.")
        return

    logger.info("[Pipeline] Building indexes...")
    build_all_indexes(lc_docs)
    logger.info("✅ Pipeline complete. Run the server now.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Legal Advisor — Data Pipeline")
    parser.add_argument("--mode", choices=["pdf", "scrape", "cache"], default="pdf")
    args = parser.parse_args()

    if args.mode == "pdf":
        pipeline_from_pdfs()
    elif args.mode == "scrape":
        pipeline_from_scrape()
    elif args.mode == "cache":
        pipeline_from_cache()
