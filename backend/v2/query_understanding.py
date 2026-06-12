"""
query_understanding.py — Intent-aware query expansion v3.

Fixes ChatGPT gap: "phone lost" → ["mobile theft IPC 379", "FIR CrPC 154"]
Also extracts law sections directly from query text.
"""
import re
from dataclasses import dataclass

# ─── Precise intent expansion ─────────────────────────────────────────────────
_INTENT_MAP: dict[str, list[str]] = {
    # Criminal
    "lost":        ["lost property police complaint CrPC 154 FIR procedure found property"],
    "kho":         ["lost property police complaint CrPC 154 FIR procedure found property"],
    "stolen":      ["theft FIR IPC 379 BNS 303 CrPC 154 stolen property police complaint"],
    "chori":       ["theft FIR IPC 379 BNS 303 CrPC 154 police complaint"],
    "fraud":       ["cheating fraud IPC 420 BNS 318 FIR criminal complaint"],
    "scam":        ["fraud cheating IPC 420 BNS 318 cyber crime IT Act 66D FIR"],
    "threat":      ["criminal intimidation IPC 506 BNS police complaint"],
    "cheat":       ["cheating IPC 420 BNS 318 FIR criminal complaint"],
    "fake":        ["fraud misrepresentation IPC 420 BNS 318 cheating"],
    "cyber":       ["cyber crime IT Act 66 66C 66D online fraud FIR cybercrime.gov.in"],
    "hack":        ["hacking IT Act 66 unauthorized access cyber crime FIR"],
    "harass":      ["harassment IPC 354 506 BNS stalking cyber harassment"],
    "blackmail":   ["extortion IPC 383 384 BNS criminal complaint FIR"],
    "arrest":      ["arrest CrPC 41 anticipatory bail CrPC 438 BNSS 482 rights"],
    "bail":        ["bail application CrPC 436 437 438 BNSS anticipatory bail sessions court"],

    # Rent / Property
    "deposit":     ["security deposit refund TPA Section 108 tenant landlord"],
    "rent":        ["rent agreement tenant landlord Transfer of Property Act 1882"],
    "evict":       ["eviction illegal TPA Section 108 tenant rights rent control"],
    "landlord":    ["landlord tenant dispute TPA Section 108 rent control act"],
    "flat":        ["tenant landlord flat rent agreement deposit TPA"],
    "bijli":       ["illegal electricity disconnection TPA Section 108 landlord tenant"],

    # Consumer
    "phone":       ["mobile phone consumer complaint Consumer Protection Act 2019 Section 35 DCDRC"],
    "product":     ["defective product consumer forum DCDRC Section 35 Consumer Protection Act 2019"],
    "refund":      ["refund consumer complaint Consumer Protection Act 2019 deficiency service"],
    "amazon":      ["ecommerce consumer complaint Consumer Protection E-Commerce Rules 2020"],
    "flipkart":    ["ecommerce consumer complaint Consumer Protection Rules 2020"],
    "delivery":    ["delivery failure consumer complaint deficiency service Consumer Protection Act"],
    "warranty":    ["warranty replacement Consumer Protection Act 2019 deficiency"],
    "defective":   ["defective goods Consumer Protection Act 2019 Section 35 DCDRC"],

    # Employment
    "fired":       ["wrongful termination Industrial Disputes Act Section 25F retrenchment compensation"],
    "salary":      ["unpaid salary Payment of Wages Act Section 4 15 labour court"],
    "job":         ["employment dispute labour court Industrial Disputes Act 1947"],
    "pf":          ["provident fund EPF Act EPFO employer contribution complaint"],
    "resign":      ["resignation notice period employment contract breach"],
    "layoff":      ["retrenchment Industrial Disputes Act 25F 25N compensation"],
    "naukri":      ["employment termination Industrial Disputes Act labour court"],

    # General
    "noise":       ["noise nuisance IPC 268 public nuisance CrPC 133 Section complaint"],
    "water":       ["water supply municipality consumer forum Public Premises Act"],
    "society":     ["housing society RWA dispute Apartment Ownership Act consumer forum"],
    "documents":   ["document recovery civil court injunction specific relief CPC"],
    "money":       ["money recovery civil suit CPC Order 37 Limitation Act 3 years"],
    "divorce":     ["divorce Hindu Marriage Act Section 13 mutual consent 13B"],
    "property":    ["property dispute TPA civil suit injunction CPC"],
    "accident":    ["motor accident MACT Motor Vehicles Act insurance claim FIR"],
    "medical":     ["medical negligence consumer forum Consumer Protection Act 2019"],
    "loan":        ["loan recovery SARFAESI Act DRT debt recovery tribunal"],
}

_ANCHORS: dict[str, str] = {
    "rent":       "Transfer of Property Act 1882 Section 108 tenant landlord deposit eviction",
    "consumer":   "Consumer Protection Act 2019 Section 35 deficiency service DCDRC forum",
    "criminal":   "IPC BNS FIR CrPC BNSS police complaint cheating theft fraud",
    "employment": "Industrial Disputes Act Section 25F Payment of Wages Act labour court",
    "general":    "Code of Civil Procedure 1908 Limitation Act legal notice civil suit",
}

_NOISE = {"please","hello","hi","kindly","sir","madam","actually","basically",
          "just","want","need","help","me","tell","kya","mujhe","mere","mera",
          "karo","batao","chahiye","hai","hain","nahi","nhi","tha","the","thi",
          "aur","or","bhi","toh","to","se","ke","ka","ki","ne","par","pe"}

_AMOUNT_RE  = re.compile(r"(?:rs\.?|rupees?|₹)\s*([\d,]+)", re.I)
_SECTION_RE = re.compile(r"section\s+(\d+[\w.-]*)", re.I)
_ACT_RE     = re.compile(r"([A-Z][a-zA-Z\s]+(?:Act|Code),\s*\d{4})")
_IPC_RE     = re.compile(r"\b(?:ipc|bns|bnss|crpc)\s*(\d+[\w.-]*)", re.I)


@dataclass
class QueryContext:
    original: str
    cleaned: str
    case_type: str
    entities: dict
    primary_query: str
    metadata_filters: dict
    sub_queries: list[str]
    intent_keywords: list[str]


def _extract_entities(text: str) -> dict:
    e: dict = {}
    if a := _AMOUNT_RE.findall(text):
        e["amounts"] = [x.replace(",", "") for x in a]
    if s := _SECTION_RE.findall(text):
        e["sections"] = s
    if ac := _ACT_RE.findall(text):
        e["acts"] = ac
    if i := _IPC_RE.findall(text):
        e["ipc_sections"] = i
    return e


def _clean(query: str) -> str:
    return " ".join(w for w in query.lower().split()
                    if w not in _NOISE and len(w) > 1)


def _expand_intent(query: str) -> list[str]:
    q = query.lower()
    expansions = []
    for trigger, terms in _INTENT_MAP.items():
        if trigger in q:
            expansions.extend(terms)
    return expansions[:3]


def _build_primary(cleaned: str, case_type: str,
                   entities: dict, expansions: list[str]) -> str:
    parts = [cleaned]
    if "sections" in entities:
        parts.append("section " + " ".join(entities["sections"]))
    if "ipc_sections" in entities:
        parts.append("IPC " + " IPC ".join(entities["ipc_sections"]))
    if "acts" in entities:
        parts.extend(entities["acts"])
    # Category anchor — critical for precision
    if case_type in _ANCHORS:
        parts.append(_ANCHORS[case_type])
    if expansions:
        parts.append(expansions[0])
    return " ".join(parts)


def _decompose(query: str) -> list[str]:
    parts = re.split(r"\band\b|\balso\b|\bmoreover\b|\baur\b|\bplus\b",
                     query, flags=re.I)
    return [p.strip() for p in parts if len(p.strip()) > 15] if len(parts) > 1 else []


def understand_query(query: str, case_type: str = "unknown") -> QueryContext:
    """Intent-aware query understanding. <2ms. No LLM."""
    cleaned    = _clean(query)
    entities   = _extract_entities(query)
    expansions = _expand_intent(query)
    filters: dict = {}
    if case_type != "unknown":
        filters["category"] = case_type
    if "acts" in entities:
        filters["law_name_hint"] = entities["acts"][0]
    return QueryContext(
        original=query, cleaned=cleaned, case_type=case_type,
        entities=entities,
        primary_query=_build_primary(cleaned, case_type, entities, expansions),
        metadata_filters=filters,
        sub_queries=_decompose(query),
        intent_keywords=expansions,
    )
