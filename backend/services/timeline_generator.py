"""Legal timeline generator — LLM with static fallback."""
import json
import re
import logging
from backend.core.llm import call_llm
from backend.core.prompts import TIMELINE_SYSTEM, TIMELINE_USER

logger = logging.getLogger(__name__)

_FALLBACK = {
    "rent": [
        {"phase":"Pre-Litigation","title":"Send Legal Notice","description":"Send registered AD legal notice to landlord demanding deposit refund within 15 days.","estimated_days_from_start":0,"is_critical":True},
        {"phase":"Pre-Litigation","title":"Wait for Response","description":"15-day compliance window.","estimated_days_from_start":15,"is_critical":True},
        {"phase":"Litigation","title":"File in Rent/Civil Court","description":"File summary suit under CPC or state rent control act.","estimated_days_from_start":30,"is_critical":True},
        {"phase":"Litigation","title":"First Hearing","description":"Court issues notice to opposite party.","estimated_days_from_start":60,"is_critical":False},
        {"phase":"Resolution","title":"Decree / Settlement","description":"Court passes decree for recovery.","estimated_days_from_start":180,"is_critical":True},
    ],
    "consumer": [
        {"phase":"Pre-Litigation","title":"File Company Complaint","description":"Formal complaint to company grievance cell.","estimated_days_from_start":0,"is_critical":True},
        {"phase":"Pre-Litigation","title":"Await Resolution","description":"30-day window under Consumer Protection Act 2019.","estimated_days_from_start":30,"is_critical":True},
        {"phase":"Filing","title":"File at DCDRC","description":"File before District Commission if claim < ₹50L.","estimated_days_from_start":45,"is_critical":True},
        {"phase":"Resolution","title":"Order / Award","description":"Commission passes order for refund/compensation.","estimated_days_from_start":180,"is_critical":True},
    ],
}


def generate_timeline(case_type: str, facts: str, outcome: str) -> list[dict]:
    try:
        raw = call_llm(
            messages=[
                {"role": "system", "content": TIMELINE_SYSTEM},
                {"role": "user", "content": TIMELINE_USER.format(
                    case_type=case_type, facts=facts, outcome=outcome)},
            ],
            max_tokens=1024,
        )
        raw = raw.strip().replace("```json", "").replace("```", "")
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            return json.loads(m.group())
        raise ValueError("No JSON array")
    except Exception as e:
        logger.warning(f"[Timeline] LLM failed, using fallback: {e}")
        return _FALLBACK.get(case_type, _FALLBACK["rent"])
