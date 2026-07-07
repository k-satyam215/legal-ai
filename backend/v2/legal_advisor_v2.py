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
from backend.v2.chat_engine import _detect_intent

logger = logging.getLogger(__name__)

# ─── Known-intent quick templates ─────────────────────────────────────────────
# Mirrors chat_engine._QUICK_CARDS content (same source of truth, structured
# for the Quick Analysis card instead of markdown). Bypasses RAG+LLM entirely
# for these well-understood patterns, avoiding misclassification-driven
# retrieval noise (e.g. 'general' anchor pulling in unrelated case law).
_STRUCTURED_QUICK: dict[str, dict] = {
    "phone_lost": {
        "issue": "Mobile phone lost (not stolen)",
        "case_type": "general",
        "laws": ["CrPC Section 154 (FIR)", "DoT CEIR Portal"],
        "analysis": "Loss of a mobile phone requires an FIR to be filed under CrPC Section 154, which enables tracing and blocking of the device via IMEI. The DoT's CEIR portal allows IMEI blocking to prevent misuse.",
        "steps": [
            "File FIR at the nearest police station under CrPC Section 154",
            "Block IMEI at ceir.gov.in (Department of Telecommunications)",
            "Contact your carrier to block the SIM card",
        ],
        "risk_level": "LOW — property loss without criminal element",
        "strategy": "Report promptly to police and telecom provider to prevent misuse of the device.",
        "notice_applicable": False,
        "follow_up_questions": ["What documents are required to file an FIR?", "Can I track my phone after filing the FIR?"],
    },
    "phone_stolen": {
        "issue": "Mobile phone stolen",
        "case_type": "criminal",
        "laws": ["IPC Section 379 / BNS Section 303", "CrPC Section 154"],
        "analysis": "Mobile phone theft is a cognizable offence under IPC Section 379, requiring mandatory FIR registration under CrPC Section 154. Police are legally bound to register the FIR and investigate.",
        "steps": [
            "File FIR immediately — IPC Section 379 / BNS Section 303",
            "Block IMEI: ceir.gov.in and contact carrier for SIM block",
            "Change all banking and account passwords immediately",
        ],
        "risk_level": "MEDIUM — financial loss risk if banking apps were accessed",
        "strategy": "File FIR immediately and secure linked financial accounts.",
        "notice_applicable": False,
        "follow_up_questions": ["What if police refuse to register the FIR?", "How do I claim insurance for the stolen phone?"],
    },
    "deposit_refund": {
        "issue": "Landlord not returning security deposit",
        "case_type": "rent",
        "laws": ["Transfer of Property Act 1882, Section 108"],
        "analysis": "Under Section 108 of the Transfer of Property Act 1882, a landlord is legally obligated to return the security deposit upon vacation of premises, subject to legitimate deductions. Failure to do so constitutes a breach of tenancy rights.",
        "steps": [
            "Send written demand via WhatsApp and email (creates documentary evidence)",
            "Issue Registered AD legal notice citing TPA Section 108",
            "File recovery suit in Rent Control Court or Civil Court",
        ],
        "risk_level": "MEDIUM — prolonged litigation if no written agreement",
        "strategy": "Escalate via written notice before pursuing court action.",
        "notice_applicable": True,
        "follow_up_questions": ["What deductions can the landlord legally make from the deposit?", "How do I draft a legal notice for deposit refund?"],
    },
    "consumer_complaint": {
        "issue": "Deficient goods/service — consumer complaint",
        "case_type": "consumer",
        "laws": ["Consumer Protection Act 2019, Section 35"],
        "analysis": "Under Section 35 of the Consumer Protection Act 2019, any consumer can file a complaint before the District Consumer Disputes Redressal Commission for deficiency of service or defective goods. The complainant is entitled to refund, replacement, and compensation.",
        "steps": [
            "File written complaint with company's grievance officer (mandatory first step)",
            "Wait 30 days — if unresolved, file at consumerhelpline.gov.in",
            "File complaint at DCDRC for claims up to ₹50 lakhs",
        ],
        "risk_level": "LOW — strong consumer protection legislation",
        "strategy": "Exhaust grievance officer route first, then escalate to DCDRC.",
        "notice_applicable": True,
        "follow_up_questions": ["What documents are required for the consumer complaint?", "What compensation can I claim apart from refund?"],
    },
    "salary_unpaid": {
        "issue": "Unpaid salary / wages",
        "case_type": "employment",
        "laws": ["Payment of Wages Act 1936, Section 15", "Industrial Disputes Act 1947"],
        "analysis": "Under Section 15 of the Payment of Wages Act 1936, an employee can file a claim before the Payment of Wages Authority for unpaid wages. Additionally, a complaint to the Labour Commissioner can initiate departmental action against the employer.",
        "steps": [
            "Send formal written demand to HR and management via email",
            "File complaint with Labour Commissioner (state-specific office)",
            "File claim under Payment of Wages Act Section 15 before the authority",
        ],
        "risk_level": "MEDIUM — depends on employment classification and documentation",
        "strategy": "Send written demand first, then escalate to Labour Commissioner.",
        "notice_applicable": True,
        "follow_up_questions": ["Am I eligible if I was employed on contract basis?", "What interest or penalty can I claim on delayed wages?"],
    },
    "fir_refused": {
        "issue": "Police refusing to register FIR",
        "case_type": "criminal",
        "laws": ["CrPC Section 154", "CrPC Section 156(3)"],
        "analysis": "Under CrPC Section 154, registration of FIR for cognizable offences is mandatory and police cannot refuse. If refused, Section 156(3) CrPC enables a Magistrate to direct investigation.",
        "steps": [
            "Submit written complaint to SP/DCP of the district",
            "File application before Magistrate under CrPC Section 156(3)",
            "File online complaint at cybercrime.gov.in for cyber-related matters",
        ],
        "risk_level": "HIGH — delay in FIR can affect evidence and investigation",
        "strategy": "Escalate to SP/DCP in writing, then Magistrate if still refused.",
        "notice_applicable": False,
        "follow_up_questions": ["What is the procedure for filing a Section 156(3) application?", "Can I file an FIR at any police station or only the local one?"],
    },
    "eviction": {
        "issue": "Forceful/illegal eviction by landlord",
        "case_type": "rent",
        "laws": ["Transfer of Property Act 1882, Section 108", "IPC Section 441"],
        "analysis": "Under TPA Section 108, a landlord cannot evict a tenant without following due legal process, and any forceful eviction or disconnection of utilities constitutes an offence under IPC Section 441 (criminal trespass).",
        "steps": [
            "File police complaint — IPC Section 441 for criminal trespass",
            "File urgent application at Rent Control Court for stay of eviction",
            "Document all communication and threats as evidence",
        ],
        "risk_level": "HIGH — immediate action required to prevent illegal eviction",
        "strategy": "File police complaint and seek urgent stay order in parallel.",
        "notice_applicable": True,
        "follow_up_questions": ["What is the legal eviction procedure a landlord must follow?", "Can I claim damages for illegal eviction attempt?"],
    },
    "cyber_fraud": {
        "issue": "Cyber/online financial fraud",
        "case_type": "criminal",
        "laws": ["IT Act 2000 Section 66D", "IPC Section 420 / BNS Section 318"],
        "analysis": "Cyber fraud involving financial loss is punishable under IT Act Section 66D (cheating by impersonation online) and IPC Section 420 (cheating). Immediate reporting to cybercrime.gov.in and the bank is critical for fund recovery.",
        "steps": [
            "Report immediately at cybercrime.gov.in (National Cybercrime Reporting Portal)",
            "Call your bank immediately — request transaction hold or reversal",
            "File FIR at local police station citing IT Act Section 66D",
        ],
        "risk_level": "HIGH — funds recovery depends on speed of reporting",
        "strategy": "Report to bank and cybercrime portal within the first hour for best recovery odds.",
        "notice_applicable": False,
        "follow_up_questions": ["What evidence should I preserve for the cyber fraud complaint?", "Is there a time limit to report and recover funds?"],
    },
}

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
    # Known-pattern fast path — bypasses RAG+LLM for well-understood intents,
    # avoiding retrieval noise when classification is weak/ambiguous.
    intent = _detect_intent(query)
    if intent and intent in _STRUCTURED_QUICK:
        data = dict(_STRUCTURED_QUICK[intent])
        return _validate(data, data["case_type"], docs=[], is_deep=False)
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
