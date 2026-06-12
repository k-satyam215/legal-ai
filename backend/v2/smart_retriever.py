"""
smart_retriever.py v4 — MMR + law-section priority + intent expansion.
"""
import re, time, hashlib, logging
from collections import defaultdict
from backend.rag.retriever import retrieve
from backend.v2.query_understanding import QueryContext

logger = logging.getLogger(__name__)

FAISS_TOP_K = 8
SUB_TOP_K   = 3
FINAL_TOP_K = 3
MAX_CONTEXT = 900
MIN_LEN     = 80

W_FAISS = 0.50
W_TFIDF = 0.25
W_META  = 0.25

_NOISE_RE   = re.compile(r"(table\s+of\s+contents|short\s+title|this\s+act\s+may\s+be\s+called|AIR\s+\d{4}|SCR\s+\d{4})", re.I)
_SECTION_RE = re.compile(r"(section\s+\d+[a-z]?|article\s+\d+|ipc\s+\d+|bnss?\s+\d+|crpc\s+\d+|rule\s+\d+)", re.I)
_SIGNAL_RE  = re.compile(r"(shall\s+be\s+liable|punishable|right\s+to|court\s+held|compensation|damages|injunction|FIR|bail|consumer\s+forum|labour\s+court|retrenchment)", re.I)


def _tokenize(t: str) -> list[str]: return re.findall(r"[a-z]+", t.lower())

def _tfidf(q_tok: list[str], doc: str) -> float:
    dt = _tokenize(doc)
    if not dt: return 0.0
    freq: dict = defaultdict(int)
    for t in dt: freq[t] += 1
    n = len(dt)
    return sum((freq.get(qt,0)/n)*(1.0 if (freq.get(qt,0)/n)<0.05 else 0.5) for qt in set(q_tok))

def _dedup(docs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for d in docs:
        h = hashlib.md5(d.get("text","")[:150].encode()).hexdigest()
        if h not in seen: seen.add(h); out.append(d)
    return out

def _boost(doc: dict, ctx: QueryContext) -> float:
    b = 0.0; meta = doc.get("metadata",{}); text = doc.get("text","")
    if meta.get("category") == ctx.case_type: b += 0.22
    hint = ctx.metadata_filters.get("law_name_hint","")
    if hint and hint.lower()[:8] in meta.get("law_name","").lower(): b += 0.12
    for sec in ctx.entities.get("sections",[]): 
        if f"section {sec}" in text.lower(): b += 0.10; break
    sec_count = len(_SECTION_RE.findall(text))
    b += min(sec_count * 0.05, 0.20)
    b += min(len(_SIGNAL_RE.findall(text)) * 0.03, 0.09)
    if _NOISE_RE.search(text): b -= 0.20
    if len(text) < MIN_LEN: b -= 0.25
    return b

def _normalize(docs: list[dict]) -> list[dict]:
    if not docs: return docs
    vals = [d["score"] for d in docs]; mn, mx = min(vals), max(vals); rng = mx-mn or 1.0
    for d in docs: d["score_norm"] = (d["score"]-mn)/rng
    return docs

def _mmr(docs: list[dict], k: int, lam: float = 0.7) -> list[dict]:
    if len(docs) <= k: return docs
    selected = [docs[0]]; candidates = docs[1:]
    while len(selected) < k and candidates:
        best_s, best_d = -float("inf"), None
        for cand in candidates:
            rel = cand.get("final_score", 0)
            ct  = set(_tokenize(cand.get("text","")[:200]))
            max_sim = max((len(ct & set(_tokenize(s.get("text","")[:200]))) / len(ct | set(_tokenize(s.get("text","")[:200]))) if ct | set(_tokenize(s.get("text","")[:200])) else 0) for s in selected)
            score = lam * rel - (1-lam) * max_sim
            if score > best_s: best_s = score; best_d = cand
        if best_d: selected.append(best_d); candidates.remove(best_d)
    return selected

def smart_retrieve(ctx: QueryContext, final_k: int = FINAL_TOP_K) -> list[dict]:
    t0   = time.perf_counter()
    docs = retrieve(ctx.primary_query, top_k=FAISS_TOP_K)
    for sq in ctx.sub_queries[:1]: docs.extend(retrieve(sq, top_k=SUB_TOP_K))
    docs = _dedup(docs); docs = _normalize(docs)
    qt   = _tokenize(ctx.primary_query)
    for d in docs:
        tf = _tfidf(qt, d.get("text",""))
        mb = _boost(d, ctx)
        d["final_score"] = W_FAISS*d.get("score_norm",0) + W_TFIDF*min(tf*10,1.0) + W_META*max(min(mb+0.5,1.0),0.0)
    docs.sort(key=lambda d: d["final_score"], reverse=True)
    result = _mmr(docs, final_k)
    logger.info(f"[Retriever] '{ctx.original[:40]}' → {len(result)} docs {(time.perf_counter()-t0)*1000:.0f}ms")
    return result

def build_optimized_context(docs: list[dict], max_chars: int = MAX_CONTEXT) -> str:
    # Sort: law section chunks first
    docs_s = sorted(docs, key=lambda d: len(_SECTION_RE.findall(d.get("text",""))), reverse=True)
    parts, used = [], 0
    for i, doc in enumerate(docs_s, 1):
        meta = doc.get("metadata",{}); text = doc.get("text","").strip()
        if not text or len(text) < MIN_LEN: continue
        snippet = text[:380]; lp = snippet.rfind(".")
        snippet = snippet[:lp+1] if lp > 150 else snippet[:300]
        law = meta.get("law_name",""); sec = meta.get("section","")
        hdr = f"[{law}{', '+sec if sec and sec not in ('General','Full Act','Full Judgment') else ''}]"
        block = f"{hdr} {snippet}"
        if used + len(block) > max_chars:
            rem = max_chars - used
            if rem > 80: parts.append(block[:rem])
            break
        parts.append(block); used += len(block) + 1
    return " | ".join(parts) if parts else "No relevant legal context found."
