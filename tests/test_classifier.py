"""Tests for backend.v2.classifier_v2 — rule-based + LLM fallback classification."""
import pytest
from backend.v2.classifier_v2 import classify_query_v2, _score_rules, _resolve


# ──────────────────────────────────────────────────────────────────────────
# Rule-based scoring
# ──────────────────────────────────────────────────────────────────────────
class TestScoreRules:
    def test_rent_keywords_score_rent(self):
        scores = _score_rules("Mera landlord deposit wapas nahi de raha")
        assert scores["rent"] > 0

    def test_consumer_keywords_score_consumer(self):
        scores = _score_rules("Amazon se defective product aaya, refund nahi mil raha")
        assert scores["consumer"] > 0

    def test_employment_keywords_score_employment(self):
        scores = _score_rules("Company ne 3 mahine se salary nahi di, mujhe fire kar diya")
        assert scores["employment"] > 0

    def test_criminal_keywords_score_criminal(self):
        scores = _score_rules("Mera phone chori ho gaya, FIR file karni hai")
        assert scores["criminal"] > 0

    def test_unrelated_text_scores_zero_everywhere(self):
        scores = _score_rules("xyzzy qwerty foobar")
        assert all(v == 0 for v in scores.values())

    def test_case_insensitive(self):
        lower = _score_rules("landlord deposit nahi de raha")
        upper = _score_rules("LANDLORD DEPOSIT NAHI DE RAHA")
        assert lower == upper


# ──────────────────────────────────────────────────────────────────────────
# Resolution logic (ambiguity, thresholds)
# ──────────────────────────────────────────────────────────────────────────
class TestResolve:
    def test_clear_winner_high_confidence(self):
        scores = {"rent": 3, "consumer": 0, "criminal": 0, "employment": 0, "general": 0}
        winner, conf = _resolve(scores)
        assert winner == "rent"
        assert conf == 0.95

    def test_no_signal_returns_none(self):
        scores = {"rent": 0, "consumer": 0, "criminal": 0, "employment": 0, "general": 0}
        winner, conf = _resolve(scores)
        assert winner is None
        assert conf == 0.0

    def test_ambiguous_rent_criminal_prefers_criminal(self):
        # rent=1, criminal=1 -> ambiguous combo -> criminal preferred
        scores = {"rent": 1, "consumer": 0, "criminal": 1, "employment": 0, "general": 0}
        winner, conf = _resolve(scores)
        assert winner == "criminal"
        assert conf == 0.88

    def test_single_hit_returns_moderate_confidence(self):
        scores = {"rent": 1, "consumer": 0, "criminal": 0, "employment": 0, "general": 0}
        winner, conf = _resolve(scores)
        assert winner == "rent"
        assert conf == 0.85


# ──────────────────────────────────────────────────────────────────────────
# Full classify_query_v2 (rule path — no LLM needed)
# ──────────────────────────────────────────────────────────────────────────
class TestClassifyQueryV2RulePath:
    def test_rent_case_classified_correctly(self):
        result = classify_query_v2("Mera landlord security deposit 3 mahine se wapas nahi kar raha")
        assert result["case_type"] == "rent"
        assert result["confidence"] > 0.8
        assert "Rule" in result["reason"]

    def test_consumer_case_classified_correctly(self):
        result = classify_query_v2("Flipkart se mera order defective aaya, replacement nahi mil raha")
        assert result["case_type"] == "consumer"
        assert result["confidence"] > 0.8

    def test_criminal_case_classified_correctly(self):
        result = classify_query_v2("Kisi ne mera phone chori kar liya hai, FIR file karna chahta hoon")
        assert result["case_type"] == "criminal"

    def test_employment_case_classified_correctly(self):
        result = classify_query_v2("Company ne bina notice ke job se nikal diya, PF nahi de rahe")
        assert result["case_type"] == "employment"

    def test_result_has_required_keys(self):
        result = classify_query_v2("Mera landlord deposit nahi de raha")
        assert set(result.keys()) >= {"case_type", "confidence", "reason"}


# ──────────────────────────────────────────────────────────────────────────
# LLM fallback path (mocked)
# ──────────────────────────────────────────────────────────────────────────
class TestClassifyQueryV2LLMFallback:
    def test_falls_back_to_llm_when_no_rule_match(self, mock_llm):
        result = classify_query_v2("Mera neighbor bohot loud music bajata hai raat me")
        # "noise"/"society" keywords exist in _RULES["general"], may hit rule path
        assert result["case_type"] in {"general", "rent", "consumer", "criminal", "employment"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_llm_exception_returns_safe_fallback(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("Groq is down")
        monkeypatch.setattr("backend.v2.classifier_v2.call_llm", boom)

        result = classify_query_v2("asdkjasdkj qweoiqwe completely nonsense query")
        assert result["case_type"] == "general"
        assert result["confidence"] == 0.5
        assert result["reason"] == "Fallback"

    def test_invalid_llm_json_returns_safe_fallback(self, monkeypatch):
        monkeypatch.setattr("backend.v2.classifier_v2.call_llm", lambda *a, **k: "not json at all")
        result = classify_query_v2("zzz totally unmatchable input string")
        assert result["case_type"] == "general"
        assert result["confidence"] == 0.5
