"""classifier_v2.py — Score-based + Hindi/Hinglish + LLM fallback."""
import json, re, logging
from backend.core.llm import call_llm
from backend.core.config import CASE_TYPES
from backend.core.prompts import CLASSIFIER_SYSTEM, CLASSIFIER_USER
logger = logging.getLogger(__name__)

_RULES: dict[str, list[str]] = {
    "rent":       ["rent","landlord","tenant","deposit","evict","lease","flat","security deposit",
                   "makan malik","kiraya","bhaada","bijli band","forceful eviction","makan khali"],
    "consumer":   ["product","flipkart","amazon","refund","defective","delivery","warranty",
                   "replacement","shopping","online order","wrong product","broken","ecommerce",
                   "myntra","meesho","nykaa","service center"],
    "employment": ["salary","job","fired","termination","employer","wages","office","company",
                   "hr","payslip","resign","retrenchment","layoff","notice period","pf",
                   "provident fund","gratuity","naukri","kaam se nikala"],
    "criminal":   ["fraud","cheat","scam","threat","police","crime","fir","arrest","bail",
                   "extortion","blackmail","fake","cheating","thagi","dhoka","farzi","thana",
                   "dhamki","paisa le gaya","chori","stolen","theft","hack","cyber","otp"],
    "general":    ["noise","neighbour","disturbance","nuisance","neighbor","sound","society",
                   "water supply","maintenance","rwa","parking","boundary","documents"],
}
_AMBIGUOUS = [
    ({"rent","criminal"},"criminal"),
    ({"consumer","criminal"},"criminal"),
    ({"employment","criminal"},"criminal"),
]

def _score_rules(query: str) -> dict[str, int]:
    q = query.lower()
    return {cat: sum(1 for kw in kws if kw in q) for cat, kws in _RULES.items()}

def _resolve(scores: dict[str, int]) -> tuple[str | None, float]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] == 0: return None, 0.0
    top, top_s = ranked[0]
    second_s   = ranked[1][1] if len(ranked) > 1 else 0
    if top_s >= 1 and (top_s - second_s) >= 2: return top, 0.95
    if top_s > 0 and second_s > 0:
        top_two = {top, ranked[1][0]}
        for combo, preferred in _AMBIGUOUS:
            if top_two == combo: return preferred, 0.88
    if top_s == 1: return top, 0.85
    return None, 0.0

def classify_query_v2(query: str) -> dict:
    scores = _score_rules(query)
    winner, conf = _resolve(scores)
    if winner: return {"case_type":winner,"confidence":conf,"reason":f"Rule: {scores[winner]} hits"}
    try:
        raw  = call_llm(messages=[{"role":"system","content":CLASSIFIER_SYSTEM},
                                   {"role":"user","content":CLASSIFIER_USER.format(query=query)}],
                        temperature=0.0, max_tokens=80)
        raw  = raw.strip().replace("```json","").replace("```","")
        m    = re.search(r"\{[\s\S]*\}", raw)
        if not m: raise ValueError("No JSON")
        data = json.loads(m.group())
        if data.get("case_type") not in CASE_TYPES: data["case_type"] = "general"
        data["confidence"] = min(float(data.get("confidence",0.75)), 0.85)
        return data
    except Exception as e:
        logger.error(f"[Classifier] {e}")
        return {"case_type":"general","confidence":0.5,"reason":"Fallback"}
