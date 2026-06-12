"""Tests for backend.v2.legal_advisor_v2 — validation, fallbacks, section enforcement."""
import pytest
from backend.v2.legal_advisor_v2 import (
    get_legal_advice_v2, get_deep_analysis,
    _enforce_sections, _normalize_risk, _grounded_laws, _validate, _parse,
    _FALLBACK, _DEEP_FALLBACK, _REQ_STD, _REQ_DEEP,
)


# ──────────────────────────────────────────────────────────────────────────
# JSON parsing
# ──────────────────────────────────────────────────────────────────────────
class TestParse:
    def test_parses_clean_json(self):
        result = _parse('{"issue": "test", "case_type": "rent"}')
        assert result["issue"] == "test"

    def test_strips_markdown_fences(self):
        result = _parse('```json\n{"issue": "test"}\n```')
        assert result["issue"] == "test"

    def test_extracts_json_from_surrounding_text(self):
        result = _parse('Here is the analysis: {"issue": "test"} thanks')
        assert result["issue"] == "test"

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            _parse("no json here at all")


# ──────────────────────────────────────────────────────────────────────────
# Section enforcement (hard rule)
# ──────────────────────────────────────────────────────────────────────────
class TestEnforceSections:
    def test_keeps_laws_with_section_numbers(self):
        data = {"laws": ["Transfer of Property Act 1882, Section 108"]}
        result = _enforce_sections(data)
        assert result["laws"] == ["Transfer of Property Act 1882, Section 108"]

    def test_flags_laws_without_sections(self):
        data = {"laws": ["Some Act with no numbers anywhere"]}
        result = _enforce_sections(data)
        assert "Insufficient" in result["laws"][0]

    def test_empty_laws_list_flagged(self):
        data = {"laws": []}
        result = _enforce_sections(data)
        assert result["laws"] == ["Insufficient legal context"]


# ──────────────────────────────────────────────────────────────────────────
# Risk normalization
# ──────────────────────────────────────────────────────────────────────────
class TestNormalizeRisk:
    def test_well_formed_risk_passes_through(self):
        assert _normalize_risk("HIGH — urgent action needed") == "HIGH — urgent action needed"

    def test_bare_level_word_gets_reason(self):
        assert _normalize_risk("LOW") == "LOW — see analysis"

    def test_non_string_returns_default(self):
        assert _normalize_risk(None) == "MEDIUM — unable to assess"
        assert _normalize_risk(123) == "MEDIUM — unable to assess"

    def test_unrecognized_format_gets_medium_prefix(self):
        result = _normalize_risk("some weird risk description here")
        assert result.startswith("MEDIUM —")


# ──────────────────────────────────────────────────────────────────────────
# Grounded laws extraction
# ──────────────────────────────────────────────────────────────────────────
class TestGroundedLaws:
    def test_extracts_law_and_section_from_metadata(self, mock_retriever_rent):
        from tests.conftest import MOCK_DOCS_RENT
        result = _grounded_laws(MOCK_DOCS_RENT)
        assert any("Transfer of Property Act 1882" in l for l in result)
        assert any("Section 108" in l for l in result)

    def test_empty_docs_returns_empty(self):
        assert _grounded_laws([]) == []

    def test_caps_at_five_entries(self):
        docs = [{"text": f"Section {i}", "metadata": {"law_name": f"Act {i}", "section": f"Section {i}"}} for i in range(10)]
        result = _grounded_laws(docs)
        assert len(result) <= 5

    def test_unknown_law_name_skipped(self):
        docs = [{"text": "no useful content", "metadata": {"law_name": "Unknown Law", "section": ""}}]
        result = _grounded_laws(docs)
        assert result == []


# ──────────────────────────────────────────────────────────────────────────
# _validate — required fields, normalization
# ──────────────────────────────────────────────────────────────────────────
class TestValidate:
    def test_fills_missing_required_fields_standard(self):
        data = {}
        result = _validate(data, "rent", [], is_deep=False)
        assert _REQ_STD <= set(result.keys())

    def test_fills_missing_required_fields_deep(self):
        data = {}
        result = _validate(data, "rent", [], is_deep=True)
        assert _REQ_DEEP <= set(result.keys())

    def test_invalid_case_type_replaced(self):
        data = {"case_type": "not_a_real_type"}
        result = _validate(data, "rent", [], is_deep=False)
        assert result["case_type"] == "rent"

    def test_invalid_case_type_falls_to_general_if_no_hint(self):
        data = {"case_type": "bogus"}
        result = _validate(data, "bogus_too", [], is_deep=False)
        assert result["case_type"] == "general"

    def test_steps_capped_at_three_for_standard(self):
        data = {"steps": ["a", "b", "c", "d", "e"]}
        result = _validate(data, "rent", [], is_deep=False)
        assert len(result["steps"]) <= 3

    def test_steps_capped_at_four_for_deep(self):
        data = {"steps": ["a", "b", "c", "d", "e"]}
        result = _validate(data, "rent", [], is_deep=True)
        assert len(result["steps"]) <= 4

    def test_follow_up_questions_capped_at_two(self):
        data = {"follow_up_questions": ["q1", "q2", "q3", "q4"]}
        result = _validate(data, "rent", [], is_deep=False)
        assert len(result["follow_up_questions"]) <= 2

    def test_grounded_laws_used_when_few_laws_provided(self, ):
        from tests.conftest import MOCK_DOCS_RENT
        data = {"laws": []}
        result = _validate(data, "rent", MOCK_DOCS_RENT, is_deep=False)
        assert len(result["laws"]) > 0

    def test_analysis_trimmed_to_two_sentences_standard(self):
        data = {"analysis": "One. Two. Three. Four."}
        result = _validate(data, "rent", [], is_deep=False)
        assert result["analysis"].count(".") <= 2

    def test_notice_applicable_coerced_to_bool(self):
        data = {"notice_applicable": "yes"}
        result = _validate(data, "rent", [], is_deep=False)
        assert isinstance(result["notice_applicable"], bool)


# ──────────────────────────────────────────────────────────────────────────
# Full pipeline (mocked LLM + retriever)
# ──────────────────────────────────────────────────────────────────────────
class TestGetLegalAdviceV2:
    def test_returns_valid_structure(self, mock_llm, mock_retriever_rent):
        result = get_legal_advice_v2("Landlord deposit nahi de raha", case_type="rent")
        assert _REQ_STD <= set(result.keys())
        assert result["case_type"] == "rent"
        assert len(result["laws"]) > 0

    def test_falls_back_gracefully_on_llm_error(self, monkeypatch, mock_retriever_rent):
        def boom(*a, **k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr("backend.v2.legal_advisor_v2.call_llm", boom)

        result = get_legal_advice_v2("Landlord deposit nahi de raha", case_type="rent")
        assert result["case_type"] == "rent"
        assert "laws" in result

    def test_no_docs_returns_insufficient_context_flag(self, mock_llm, mock_retriever_empty, monkeypatch):
        # Force LLM to return laws without section numbers
        monkeypatch.setattr(
            "backend.v2.legal_advisor_v2.call_llm",
            lambda *a, **k: '{"issue":"x","case_type":"general","laws":["Some Act no number"],'
                            '"analysis":"x. y.","steps":["a"],"risk":"LOW","strategy":"s",'
                            '"notice_applicable":false,"follow_up_questions":["q1"]}'
        )
        result = get_legal_advice_v2("totally vague query", case_type="general")
        assert "Insufficient" in result["laws"][0]


class TestGetDeepAnalysis:
    def test_returns_valid_deep_structure(self, mock_llm, mock_retriever_rent):
        result = get_deep_analysis("Landlord deposit nahi de raha", case_type="rent")
        assert _REQ_DEEP <= set(result.keys())
        assert "scenario_analysis" in result
        assert "primary_law" in result

    def test_falls_back_gracefully_on_llm_error(self, monkeypatch, mock_retriever_rent):
        def boom(*a, **k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr("backend.v2.legal_advisor_v2.call_llm", boom)

        result = get_deep_analysis("Landlord deposit nahi de raha", case_type="rent")
        assert _REQ_DEEP <= set(result.keys())
        assert result["case_type"] == "rent"
