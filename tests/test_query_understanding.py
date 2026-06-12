"""Tests for backend.v2.query_understanding — intent expansion & entity extraction."""
import pytest
from backend.v2.query_understanding import (
    understand_query, _extract_entities, _clean, _expand_intent, _decompose, QueryContext,
)


class TestExtractEntities:
    def test_extracts_rupee_amount(self):
        e = _extract_entities("Landlord owes me Rs. 50,000 deposit")
        assert "amounts" in e
        assert "50000" in e["amounts"]

    def test_extracts_section_number(self):
        e = _extract_entities("File complaint under Section 108")
        assert "sections" in e
        assert "108" in e["sections"]

    def test_extracts_ipc_section(self):
        e = _extract_entities("FIR under IPC 379 for theft")
        assert "ipc_sections" in e
        assert "379" in e["ipc_sections"]

    def test_extracts_act_name(self):
        e = _extract_entities("My case falls under Consumer Protection Act, 2019")
        assert "acts" in e

    def test_no_entities_in_plain_text(self):
        e = _extract_entities("mera phone kho gaya")
        assert "amounts" not in e
        assert "sections" not in e


class TestClean:
    def test_removes_noise_words(self):
        cleaned = _clean("mujhe mera deposit chahiye please help kya karu")
        for noise in ("mujhe", "mera", "please", "kya"):
            assert noise not in cleaned.split()

    def test_lowercases(self):
        cleaned = _clean("DEPOSIT WAPAS")
        assert cleaned == cleaned.lower()


class TestExpandIntent:
    def test_lost_triggers_expansion(self):
        expansions = _expand_intent("mera phone kho gaya hai")
        assert len(expansions) > 0
        assert any("CrPC" in e or "154" in e for e in expansions)

    def test_stolen_triggers_theft_expansion(self):
        expansions = _expand_intent("mera phone chori ho gaya")
        assert any("379" in e or "theft" in e.lower() for e in expansions)

    def test_max_three_expansions(self):
        # query hitting many triggers
        expansions = _expand_intent("phone chori fraud scam cyber hack")
        assert len(expansions) <= 3

    def test_no_trigger_returns_empty(self):
        expansions = _expand_intent("xyzabc nonsense words here")
        assert expansions == []


class TestDecompose:
    def test_splits_on_and(self):
        parts = _decompose("Landlord is not returning deposit and also disconnected electricity supply")
        assert len(parts) >= 1

    def test_short_query_not_decomposed(self):
        parts = _decompose("deposit nahi mil raha")
        assert parts == []


class TestUnderstandQuery:
    def test_returns_query_context(self):
        ctx = understand_query("Landlord deposit nahi de raha, Rs 50000 hai", case_type="rent")
        assert isinstance(ctx, QueryContext)
        assert ctx.case_type == "rent"
        assert ctx.original == "Landlord deposit nahi de raha, Rs 50000 hai"

    def test_primary_query_includes_anchor_for_known_case_type(self):
        ctx = understand_query("deposit wapas nahi mil raha", case_type="rent")
        assert "Transfer of Property Act" in ctx.primary_query

    def test_metadata_filters_set_category(self):
        ctx = understand_query("deposit issue", case_type="rent")
        assert ctx.metadata_filters.get("category") == "rent"

    def test_unknown_case_type_no_category_filter(self):
        ctx = understand_query("some query", case_type="unknown")
        assert "category" not in ctx.metadata_filters

    def test_entities_extracted_into_context(self):
        ctx = understand_query("FIR under IPC 379 file karna hai", case_type="criminal")
        assert "ipc_sections" in ctx.entities
