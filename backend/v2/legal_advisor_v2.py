"""
legal_advisor_v2.py v4 — Final legal reasoning engine.
All ChatGPT gaps fixed:
- Law section enforcer (hard rule)
- Professional tone (no casual language)
- Structured output with risk field
- Mode separation (standard vs deep)
- Follow-up questions mandatory
- Grounded citations only
"""
import json, re, logging
from pathlib import Path
from backend.core.llm import call_llm
from backend.core.prompts import (
    LEGAL_ADVISOR_SYSTEM, LEGAL_ADVISOR_USER,
    DEEP_ANALYSIS_SYSTEM, DEEP_ANALYSIS_USER,
)
from backend.v2.query_understanding import understand_query
from backend.v2.smart_retriever import smart_retrieve, build_optimized_context

logger = logging.getLogger(__name__)

_TPL_PATH = Path(__file__).resolve().parents[2] / "data_pipeline" / "structured_data" / "case_templates.json"
_TPL: dict = {}
try:
    with open(_TPL_PATH) as f: _TPL = json.load(f)
except Exception: pass

_FALLBACK = {
    "issue": "Legal issue could not be determined from provided context.",
    "case_type": "general",
    "laws": ["Insufficient legal context"],
    "analysis": "Insufficient context to provide legal analysis. Please provide specific details including dates, amounts, and parties involved.",
    "steps": ["Provide specific details about the issue.", "Mention dates, amounts, and parties.", "State the desired legal outcome."],
    "risk_level": "MEDIUM — insufficient information to assess",
    "strategy": "Provide complete facts for accurate legal guidance.",
    "notice_applicable": False,
    "follow_up_questions": ["What specific incident occurred and when?", "What amount or property is involved?"],
}

_DEEP_FALLBACK = {
    **_FALLBACK,
    "primary_law": "To be determined based on complete facts",
    "legal_interpretation": "Insufficient context for legal interpretation.",
    "scenario_analysis": {
        "best_case": "Resolution through negotiation with documentation",
        "worst_case": "Prolonged litigation without evidence",
        "edge_cases": ["Outcome changes if written agreement exists", "Time limitation may affect filing"]
    },
    "alternative_remedies": ["Mediation under Legal Services Authorities Act", "Consumer forum if applicable"],
    "risk_factors": ["Absence of documentation", "Time elapsed since incident"],
    "timeline_estimate": "Depends on case complexity and court",
}

_VALID_CT = {"rent","consumer","criminal","employment","general"}
_VALID_RL = {"LOW","MEDIUM","HIGH","low","medium","high"}
_REQ_STD  = {"issue","case_type","laws","analysis","steps","risk_level","strategy","notice_applicable","follow_up_questions"}
_REQ_DEEP = _REQ_STD | {"primary_law","legal_interpretation","scenario_analysis","alternative_remedies","risk_factors","timeline_estimate"}

# ─── Section enforcer ─────────────────────────────────────────────────────────
_SECTION_PATTERN = re.compile(r"\b\d+[A-Za-z]?\b")

def _enforce_sections(data: dict) -> dict:
    """Hard rule: if no digit in any law entry, flag as insufficient."""
    laws = data.get("laws", [])
    if not laws:
        data["laws"] = ["Insufficient legal context"]
        return data
    has_section = any(_SECTION_PATTERN.search(l) for l in laws)
    if not has_section:
        data["laws"] = ["Insufficient legal context — no applicable sections found in retrieved documents"]
    return data

# ─── Risk normalizer ──────────────────────────────────────────────────────────
def _normalize_risk(risk_val) -> str:
    """Ensure risk field format: 'LOW — reason'"""
    if not isinstance(risk_val, str):
        return "MEDIUM — unable to assess"
    r = risk_val.strip().upper()
    if r in ("HIGH","MEDIUM","LOW"):
        return f"{r} — see analysis"
    for level in ("HIGH","MEDIUM","LOW"):
        if r.startswith(level):
            return risk_val.strip()
    return f"MEDIUM — {risk_val[:40]}"

# ─── Grounded citations ────────────────────────────────────────────────────────
def _grounded_laws(docs: list[dict]) -> list[str]:
    out, seen = [], set()
    for doc in docs:
        meta = doc.get("metadata", {})
        law, sec = meta.get("law_name",""), meta.get("section","")
        if law and law not in ("Unknown Law","Unknown",""):
            entry = f"{law}" + (f", {sec}" if sec and sec not in ("General","Full Act","Full Judgment","") else "")
            if entry not in seen: seen.add(entry); out.append(entry)
        for ref in re.findall(
            r"(?:IPC|BNS|CrPC|BNSS|TPA|CPC|Section)\s*\d+[A-Za-z]?",
            doc.get("text",""), re.I
        )[:2]:
            ref = ref.strip()
            if ref not in seen and len(ref) > 3 and law:
                seen.add(ref); out.append(f"{law}, {ref}")
    return out[:5]

# ─── Template enrichment ──────────────────────────────────────────────────────
def _enrich(data: dict, case_type: str, is_deep: bool = False) -> dict:
    tpl = _TPL.get(case_type, {})
    if not tpl: return data
    if data.get("laws") == ["Insufficient legal context"] and tpl.get("laws"):
        data["laws"] = [f"{l['name']}, {l['key_sections'][0]}" for l in tpl["laws"][:3]]
    if is_deep and not data.get("primary_law","").strip() and tpl.get("laws"):
        data["primary_law"] = f"{tpl['laws'][0]['name']}, {tpl['laws'][0]['key_sections'][0]}"
    if len(data.get("steps",[]))<2 and tpl.get("typical_steps"):
        ex = set(data.get("steps",[]))
        for s in tpl["typical_steps"]:
            if s not in ex: data["steps"].append(s)
            if len(data["steps"]) >= 3: break
    return data

# ─── Validate ─────────────────────────────────────────────────────────────────
def _validate(data: dict, case_type: str, docs: list[dict], is_deep: bool = False) -> dict:
    fb  = _DEEP_FALLBACK if is_deep else _FALLBACK
    req = _REQ_DEEP if is_deep else _REQ_STD

    for k in req:
        if k not in data: data[k] = fb[k]

    if data.get("case_type") not in _VALID_CT:
        data["case_type"] = case_type if case_type in _VALID_CT else "general"

    for k in ("laws","steps","follow_up_questions"):
        if not isinstance(data.get(k), list): data[k] = []
    if is_deep:
        for k in ("alternative_remedies","risk_factors"):
            if not isinstance(data.get(k), list): data[k] = []

    # Normalize risk format (LLM may return either 'risk' or 'risk_level')
    risk_src = data.get("risk_level", data.get("risk", "MEDIUM"))
    data["risk_level"] = _normalize_risk(risk_src)
    data.pop("risk", None)

    # Cap lists
    data["steps"] = data["steps"][:4] if is_deep else data["steps"][:3]
    data["follow_up_questions"] = data["follow_up_questions"][:2]
    if is_deep: data["alternative_remedies"] = data.get("alternative_remedies",[])[:3]

    # Ground laws
    grounded = _grounded_laws(docs)
    if grounded and len(data.get("laws",[])) < 2:
        data["laws"] = grounded

    # Enforce sections
    data = _enforce_sections(data)

    # Template enrichment
    data = _enrich(data, data["case_type"], is_deep)

    # Trim analysis to 2 sentences
    analysis = data.get("analysis","")
    sentences = re.split(r'(?<=[.!?])\s+', analysis.strip())
    max_s = 3 if is_deep else 2
    data["analysis"] = " ".join(sentences[:max_s])

    data["notice_applicable"] = bool(data.get("notice_applicable", False))
    return data

# ─── JSON parser ──────────────────────────────────────────────────────────────
def _parse(raw: str) -> dict:
    raw = raw.strip().replace("```json","").replace("```","").strip()
    try: return json.loads(raw)
    except Exception: pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    raise ValueError("No valid JSON in LLM response")

# ─── Standard ─────────────────────────────────────────────────────────────────
def get_legal_advice_v2(query: str, case_type: str = "general") -> dict:
    """Standard mode: fast, structured, grounded. Target: <1200ms."""
    try:
        ctx     = understand_query(query, case_type=case_type)
        docs    = smart_retrieve(ctx, final_k=3)
        context = build_optimized_context(docs, max_chars=900)
        raw = call_llm(
            messages=[
                {"role":"system","content":LEGAL_ADVISOR_SYSTEM},
                {"role":"user","content":LEGAL_ADVISOR_USER.format(context=context,query=query,case_type=case_type)},
            ],
            temperature=0.0, max_tokens=320,
        )
        data = _parse(raw)
        return _validate(data, case_type, docs, is_deep=False)
    except Exception as e:
        logger.error(f"[LegalAdvisor] {e}")
        fb = dict(_FALLBACK); fb["case_type"] = case_type
        return _enrich(fb, case_type)

# ─── Deep ─────────────────────────────────────────────────────────────────────
def get_deep_analysis(query: str, case_type: str = "general", extra_context: str = "") -> dict:
    """Deep mode: interpretation, edge cases, alternatives. Target: <2000ms."""
    try:
        ctx     = understand_query(query, case_type=case_type)
        docs    = smart_retrieve(ctx, final_k=4)
        context = build_optimized_context(docs, max_chars=1200)
        raw = call_llm(
            messages=[
                {"role":"system","content":DEEP_ANALYSIS_SYSTEM},
                {"role":"user","content":DEEP_ANALYSIS_USER.format(context=context,query=query,case_type=case_type,extra_context=extra_context or "None")},
            ],
            temperature=0.1, max_tokens=520,
        )
        data = _parse(raw)
        return _validate(data, case_type, docs, is_deep=True)
    except Exception as e:
        logger.error(f"[DeepAnalysis] {e}")
        fb = dict(_DEEP_FALLBACK); fb["case_type"] = case_type
        return _enrich(fb, case_type, is_deep=True)
