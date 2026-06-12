"""Text cleaning pipeline for scraped legal documents."""
import re
import hashlib
import logging
from html import unescape

logger = logging.getLogger(__name__)


def strip_html(text: str) -> str:
    text = unescape(text)
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_noise(text: str) -> str:
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"={5,}|-{5,}|_{5,}", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    return text


def normalize_legal_terms(text: str) -> str:
    replacements = {
        r"\bS\.C\b\.?": "Supreme Court",
        r"\bH\.C\b\.?": "High Court",
        r"\bI\.P\.C\b\.?": "IPC",
        r"\bC\.P\.C\b\.?": "CPC",
        r"\bT\.P\.A\b\.?": "Transfer of Property Act",
        r"\bA\.I\.R\b\.?": "AIR",
        r"\bSCR\b": "Supreme Court Reports",
        r"\bvs\.?\b": "vs",
        r"\bV/s\b": "vs",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_document(text: str) -> str:
    text = strip_html(text)
    text = remove_noise(text)
    text = normalize_whitespace(text)
    text = normalize_legal_terms(text)
    return text


def _doc_hash(text: str) -> str:
    return hashlib.md5(text[:500].encode()).hexdigest()


def deduplicate(docs: list[dict], text_key: str = "text") -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for doc in docs:
        text = doc.get(text_key, "")
        h = _doc_hash(text)
        if h not in seen and len(text) > 100:
            seen.add(h)
            doc[text_key] = clean_document(text)
            unique.append(doc)
    removed = len(docs) - len(unique)
    logger.info(f"[Cleaner] {len(docs)} → {len(unique)} docs ({removed} removed)")
    return unique


def clean_pipeline(docs: list[dict], text_key: str = "text") -> list[dict]:
    for doc in docs:
        if text_key in doc:
            doc[text_key] = clean_document(doc[text_key])
    docs = [d for d in docs if len(d.get(text_key, "")) >= 100]
    return deduplicate(docs, text_key=text_key)
