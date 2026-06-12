"""
Section-aware + semantic chunking for Indian legal documents.
"""
import re
import hashlib
from dataclasses import dataclass, asdict
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import CHUNK_SIZE, CHUNK_OVERLAP

SECTION_PATTERN = re.compile(
    r"(Section\s+\d+[\w.-]*|S\.\s*\d+|Article\s+\d+[\w.-]*|Rule\s+\d+)",
    re.IGNORECASE,
)
ACT_PATTERN = re.compile(r"([A-Z][a-zA-Z\s]+(?:Act|Code|Ordinance|Rules),\s*\d{4})")


@dataclass
class LegalChunk:
    text: str
    law_name: str
    section: str
    category: str
    source: str
    keywords: list[str]
    court: str
    date: str
    chunk_id: str

    def to_document(self) -> Document:
        meta = asdict(self)
        meta.pop("text")
        return Document(page_content=self.text, metadata=meta)

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_keywords(text: str) -> list[str]:
    legal_terms = [
        "section", "act", "court", "judgment", "notice", "deposit",
        "eviction", "termination", "consumer", "landlord", "tenant",
        "employer", "employee", "salary", "criminal", "bail", "FIR",
        "complaint", "damages", "injunction", "decree",
    ]
    words = set(text.lower().split())
    return [t for t in legal_terms if t in words]


def _detect_section(text: str) -> str:
    match = SECTION_PATTERN.search(text)
    return match.group(0) if match else "General"


def _detect_act(text: str) -> str:
    match = ACT_PATTERN.search(text)
    return match.group(0) if match else "Unknown Law"


def _make_chunk_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _classify_category(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["rent", "tenant", "landlord", "deposit", "eviction", "lease"]):
        return "rent"
    if any(w in t for w in ["consumer", "deficiency", "service", "goods", "refund"]):
        return "consumer"
    if any(w in t for w in ["murder", "theft", "fraud", "fir", "bail", "ipc", "bns"]):
        return "criminal"
    if any(w in t for w in ["employment", "salary", "termination", "dismissal", "employer", "workman"]):
        return "employment"
    return "general"


class SectionAwareChunker:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                 source="", law_name="", court="", date=""):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.source        = source
        self.law_name      = law_name
        self.court         = court
        self.date          = date
        self._splitter     = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, text: str) -> list[LegalChunk]:
        sections = SECTION_PATTERN.split(text)
        chunks: list[LegalChunk] = []
        current_section = "General"
        buffer = ""

        for part in sections:
            if SECTION_PATTERN.match(part):
                current_section = part.strip()
                buffer += " " + part
            else:
                buffer += " " + part

            if len(buffer) > self.chunk_size:
                for sc in self._splitter.split_text(buffer.strip()):
                    sc = sc.strip()
                    if len(sc) < 80:
                        continue
                    chunks.append(LegalChunk(
                        text=sc,
                        law_name=self.law_name or _detect_act(sc),
                        section=current_section,
                        category=_classify_category(sc),
                        source=self.source,
                        keywords=_extract_keywords(sc),
                        court=self.court,
                        date=self.date,
                        chunk_id=_make_chunk_id(sc),
                    ))
                buffer = ""

        if buffer.strip() and len(buffer.strip()) >= 80:
            for sc in self._splitter.split_text(buffer.strip()):
                sc = sc.strip()
                if len(sc) < 80:
                    continue
                chunks.append(LegalChunk(
                    text=sc,
                    law_name=self.law_name or _detect_act(sc),
                    section=current_section,
                    category=_classify_category(sc),
                    source=self.source,
                    keywords=_extract_keywords(sc),
                    court=self.court,
                    date=self.date,
                    chunk_id=_make_chunk_id(sc),
                ))
        return chunks


def chunks_to_documents(chunks: list[LegalChunk]) -> list[Document]:
    return [c.to_document() for c in chunks]
