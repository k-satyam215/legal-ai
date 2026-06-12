"""
conftest.py — Shared pytest fixtures.

Strategy:
- Mock backend.core.llm.call_llm globally so NO real Groq API calls happen.
- Mock backend.rag.retriever.retrieve so NO real FAISS index is required.
- Provide a FastAPI TestClient with the app's lifespan FAISS-loading skipped.
"""
import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────────────────────────────────
# Mock retrieval documents — used across classifier / retriever / advisor tests
# ──────────────────────────────────────────────────────────────────────────
MOCK_DOCS_RENT = [
    {
        "text": "Section 108 of the Transfer of Property Act 1882 obligates the lessor to "
                "return the security deposit upon determination of the lease, subject to "
                "lawful deductions for damages beyond normal wear and tear.",
        "metadata": {"law_name": "Transfer of Property Act 1882", "section": "Section 108", "category": "rent"},
        "score": 0.91,
    },
    {
        "text": "A tenant aggrieved by non-refund of security deposit may approach the Rent "
                "Control Court for recovery along with interest.",
        "metadata": {"law_name": "Transfer of Property Act 1882", "section": "Section 108", "category": "rent"},
        "score": 0.83,
    },
]

MOCK_DOCS_CONSUMER = [
    {
        "text": "Under Section 35 of the Consumer Protection Act 2019, a complaint may be "
                "filed before the District Consumer Disputes Redressal Commission for "
                "deficiency in service or defective goods.",
        "metadata": {"law_name": "Consumer Protection Act 2019", "section": "Section 35", "category": "consumer"},
        "score": 0.88,
    },
]

MOCK_DOCS_EMPTY: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────
# LLM mock — returns valid strict-JSON for legal advisor / deep analysis,
# and plain text for chat / classifier fallback.
# ──────────────────────────────────────────────────────────────────────────
def _fake_llm_response(messages, **kwargs):
    """Inspect the system prompt to decide what kind of JSON/text to return."""
    system_content = ""
    for m in messages:
        if m.get("role") == "system":
            system_content = m.get("content", "")
            break

    sys_lower = system_content.lower()

    # Deep analysis
    if "legal_interpretation" in sys_lower or "scenario_analysis" in sys_lower or "deep" in sys_lower:
        return json.dumps({
            "issue": "Landlord refusing to refund security deposit after lease termination.",
            "case_type": "rent",
            "primary_law": "Transfer of Property Act 1882, Section 108",
            "laws": ["Transfer of Property Act 1882, Section 108"],
            "legal_interpretation": "Section 108 imposes a statutory duty on the lessor to "
                                     "return the deposit absent lawful deductions.",
            "scenario_analysis": {
                "best_case": "Landlord refunds after legal notice.",
                "worst_case": "Tenant must file recovery suit.",
                "edge_cases": ["No written agreement exists", "Deductions claimed without proof"],
            },
            "analysis": "The landlord's continued retention of the deposit beyond a "
                        "reasonable period after lease termination breaches Section 108. "
                        "The tenant has a statutory remedy to recover the amount with interest.",
            "steps": [
                "Send a written demand via email/WhatsApp",
                "Issue a registered legal notice citing Section 108",
                "File a recovery suit in the Rent Control Court",
            ],
            "alternative_remedies": ["Mediation under Legal Services Authorities Act"],
            "risk_level": "MEDIUM",
            "risk": "MEDIUM — prolonged litigation if undocumented",
            "risk_factors": ["Absence of written rent agreement"],
            "strategy": "Send a legal notice first; escalate to Rent Control Court if ignored.",
            "timeline_estimate": "2-6 months depending on court backlog",
            "notice_applicable": True,
            "follow_up_questions": [
                "Do you have a written rental agreement?",
                "What amount was paid as security deposit?",
            ],
        })

    # Standard legal advice (classifier system prompt won't match this)
    if "case_type" in sys_lower and "follow_up_questions" in sys_lower:
        return json.dumps({
            "issue": "Landlord refusing to refund security deposit after lease termination.",
            "case_type": "rent",
            "laws": ["Transfer of Property Act 1882, Section 108"],
            "analysis": "The landlord's retention of the deposit after lease termination "
                        "breaches Section 108 of the Transfer of Property Act 1882.",
            "steps": [
                "Send a written demand via email/WhatsApp",
                "Issue a registered legal notice citing Section 108",
                "File a recovery suit in the Rent Control Court",
            ],
            "risk_level": "MEDIUM",
            "risk": "MEDIUM — prolonged litigation if undocumented",
            "strategy": "Send a legal notice first; escalate if ignored.",
            "notice_applicable": True,
            "follow_up_questions": [
                "Do you have a written rental agreement?",
                "What amount was paid as security deposit?",
            ],
        })

    # Classifier LLM fallback
    if "case_type" in sys_lower and "confidence" in sys_lower:
        return json.dumps({"case_type": "general", "confidence": 0.6, "reason": "LLM fallback classification"})

    # Chat engine — plain text reply
    return "Aapko CrPC Section 154 ke under FIR file karni chahiye. Yeh ek cognizable offence hai."


@pytest.fixture
def mock_llm():
    """Patch call_llm everywhere it is imported."""
    with patch("backend.core.llm.call_llm", side_effect=_fake_llm_response) as p1, \
         patch("backend.v2.classifier_v2.call_llm", side_effect=_fake_llm_response) as p2, \
         patch("backend.v2.legal_advisor_v2.call_llm", side_effect=_fake_llm_response) as p3, \
         patch("backend.v2.chat_engine.call_llm", side_effect=_fake_llm_response) as p4, \
         patch("backend.services.notice_generator.call_llm", side_effect=_fake_llm_response) as p5, \
         patch("backend.services.timeline_generator.call_llm", side_effect=_fake_llm_response) as p6:
        yield {"core": p1, "classifier": p2, "advisor": p3, "chat": p4, "notice": p5, "timeline": p6}


@pytest.fixture
def mock_retriever_rent():
    with patch("backend.v2.smart_retriever.retrieve", return_value=list(MOCK_DOCS_RENT)) as p:
        yield p


@pytest.fixture
def mock_retriever_consumer():
    with patch("backend.v2.smart_retriever.retrieve", return_value=list(MOCK_DOCS_CONSUMER)) as p:
        yield p


@pytest.fixture
def mock_retriever_empty():
    with patch("backend.v2.smart_retriever.retrieve", return_value=[]) as p:
        yield p


@pytest.fixture
def app_client(mock_llm, mock_retriever_rent):
    """
    FastAPI TestClient with FAISS startup skipped and LLM/retriever mocked.
    Use this for endpoint/integration tests.
    """
    with patch("backend.rag.loader.initialize", return_value=None), \
         patch("backend.rag.loader.get_index", return_value=MagicMock(ntotal=100)), \
         patch("backend.v2.memory.get_memory") as mock_mem_getter:

        # Provide a real-ish in-memory MemoryStore so chat tests work
        from backend.v2.memory import MemoryStore
        store = MemoryStore()
        mock_mem_getter.return_value = store

        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the in-process TTL cache between tests to avoid cross-test pollution."""
    from backend.core.cache import cache_clear
    cache_clear()
    yield
    cache_clear()
