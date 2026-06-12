"""Data pipeline chunker — wraps backend SectionAwareChunker."""
import logging
from langchain.docstore.document import Document
from backend.rag.chunking import SectionAwareChunker, LegalChunk, chunks_to_documents

logger = logging.getLogger(__name__)


def chunk_case_law(doc: dict) -> list[LegalChunk]:
    chunker = SectionAwareChunker(
        source=doc.get("url", ""),
        law_name=doc.get("title", "Case Law"),
        court=doc.get("court", ""),
        date=doc.get("date", ""),
    )
    return chunker.chunk(doc.get("text", ""))


def chunk_act_section(doc: dict) -> list[LegalChunk]:
    chunker = SectionAwareChunker(
        source=doc.get("source", ""),
        law_name=doc.get("law_name", "Unknown Act"),
        court="", date="",
    )
    chunks = chunker.chunk(doc.get("text", ""))
    for c in chunks:
        if doc.get("section"):
            c.section = doc["section"]
        if doc.get("category"):
            c.category = doc["category"]
    return chunks


def chunk_all_documents(docs: list[dict]) -> list[Document]:
    all_chunks: list[LegalChunk] = []
    for doc in docs:
        if "url" in doc and "citation" in doc:
            chunks = chunk_case_law(doc)
        else:
            chunks = chunk_act_section(doc)
        all_chunks.extend(chunks)
    logger.info(f"[Chunker] Total chunks: {len(all_chunks)}")
    return chunks_to_documents(all_chunks)
