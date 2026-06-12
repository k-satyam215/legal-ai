"""Tests for backend.v2.chat_engine — quick cards, intent detection, LLM chat path."""
import pytest
from backend.v2.chat_engine import chat_response, _detect_intent, _QUICK_CARDS, _INTENTS


class TestDetectIntent:
    def test_phone_lost_detected(self):
        assert _detect_intent("mera phone kho gaya hai") == "phone_lost"

    def test_phone_stolen_detected(self):
        assert _detect_intent("kisi ne mera phone chori kar liya") == "phone_stolen"

    def test_deposit_refund_detected(self):
        assert _detect_intent("security deposit wapas nahi mil raha") == "deposit_refund"

    def test_consumer_complaint_detected(self):
        assert _detect_intent("amazon se refund nahi mil raha") == "consumer_complaint"

    def test_salary_unpaid_detected(self):
        assert _detect_intent("company ne salary nahi diya 2 mahine se") == "salary_unpaid"

    def test_fir_refused_detected(self):
        assert _detect_intent("police fir nahi le rahi") == "fir_refused"

    def test_eviction_detected(self):
        assert _detect_intent("landlord force se ghar se nikal raha hai") == "eviction"

    def test_cyber_fraud_detected(self):
        assert _detect_intent("online fraud ho gaya OTP diya tha") == "cyber_fraud"

    def test_documents_detected(self):
        assert _detect_intent("kaunse documents chahiye FIR ke liye") == "documents"

    def test_no_intent_returns_none(self):
        assert _detect_intent("xyz random text with no signal words") is None

    def test_all_quick_card_intents_have_cards(self):
        for intent, _ in _INTENTS:
            assert intent in _QUICK_CARDS


class TestQuickCards:
    def test_all_cards_non_empty_strings(self):
        for intent, card in _QUICK_CARDS.items():
            assert isinstance(card, str)
            assert len(card) > 20

    def test_phone_lost_card_mentions_crpc(self):
        assert "CrPC" in _QUICK_CARDS["phone_lost"]

    def test_phone_stolen_card_mentions_ipc_379(self):
        assert "379" in _QUICK_CARDS["phone_stolen"]


class TestChatResponse:
    def test_quick_card_returned_for_known_intent(self):
        result = chat_response("mera phone kho gaya", [])
        assert result["quick_card"] == "phone_lost"
        assert result["needs_deep_advice"] is True
        assert result["detected_intent"] == "phone_lost"

    def test_quick_card_reply_matches_card_text(self):
        result = chat_response("mera phone chori ho gaya", [])
        assert result["reply"] == _QUICK_CARDS["phone_stolen"]

    def test_llm_path_for_unmatched_intent(self, mock_llm):
        result = chat_response("kya mujhe vakil ki zaroorat padegi is case me", [])
        assert result["quick_card"] is None
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 0

    def test_llm_exception_returns_safe_error_message(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr("backend.v2.chat_engine.call_llm", boom)

        result = chat_response("ek naya sawal hai jo intent match nahi karega bilkul", [])
        assert "technical error" in result["reply"].lower()
        assert result["needs_deep_advice"] is False

    def test_needs_deep_advice_for_legal_keywords(self, mock_llm):
        result = chat_response("Is case me konsi section lagti hai aur court me kya hoga", [])
        assert result["needs_deep_advice"] is True

    def test_response_has_all_required_keys(self, mock_llm):
        result = chat_response("kuch general sawal hai is bare me detail me batao please", [])
        assert set(result.keys()) == {"reply", "quick_card", "needs_deep_advice", "suggested_action", "detected_intent"}
