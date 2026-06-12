"""Tests for backend.v2.smart_retriever — MMR re-ranking, dedup, scoring, context building."""
import pytest
from backend.v2.smart_retriever import (
    smart_retrieve, build_optimized_context, _dedup, _mmr, _tfidf, _tokenize, _boost, _normalize,
)
from backend.v2.query_understanding import understand_query


SAMPLE_DOCS = [
    {"text": "Section 108 of the Transfer of Property Act 1882 governs deposit refund.",
     "metadata": {"law_name": "Transfer of Property Act 1882", "section": "Section 108", "category": "rent"},
     "score": 0.9},
    {"text": "Section 108 of the Transfer of Property Act 1882 governs deposit refund.",  # duplicate
     "metadata": {"law_name": "Transfer of Property Act 1882", "section": "Section 108", "category": "rent"},
     "score": 0.85},
    {"text": "Table of contents: Chapter 1, Chapter 2, short title this Act may be called",
     "metadata": {"law_name": "Some Act", "section": "General", "category": "rent"},
     "score": 0.5},
    {"text": "The court held that compensation and damages shall be liable under Section 108 "
             "for failure to refund the deposit within a reasonable time after lease termination.",
     "metadata": {"law_name": "Transfer of Property Act 1882", "section": "Section 108", "category": "rent"},
     "score": 0.88},
]


class TestTokenizeAndTfidf:
    def test_tokenize_lowercases_and_extracts_words(self):
        toks = _tokenize("Section 108 Transfer-of-Property!")
        assert "section" in toks
        assert "transfer" in toks

    def test_tfidf_zero_for_empty_doc(self):
        assert _tfidf(["deposit"], "") == 0.0

    def test_tfidf_positive_for_matching_terms(self):
        score = _tfidf(["deposit", "section"], "deposit refund under section 108")
        assert score > 0


class TestDedup:
    def test_removes_exact_duplicates(self):
        result = _dedup(SAMPLE_DOCS)
        texts = [d["text"][:50] for d in result]
        assert len(texts) == len(set(texts))

    def test_keeps_distinct_docs(self):
        result = _dedup(SAMPLE_DOCS)
        assert len(result) == 3  # one duplicate removed from 4


class TestNormalize:
    def test_normalize_adds_score_norm(self):
        docs = [{"score": 0.5}, {"score": 1.0}]
        result = _normalize(docs)
        assert all("score_norm" in d for d in result)
        assert result[0]["score_norm"] == 0.0
        assert result[1]["score_norm"] == 1.0

    def test_normalize_empty_list(self):
        assert _normalize([]) == []


class TestBoost:
    def test_section_pattern_increases_boost(self):
        ctx = understand_query("deposit refund issue", case_type="rent")
        doc_with_section = {"text": "Section 108 governs this matter.", "metadata": {"category": "rent"}}
        doc_without = {"text": "Some unrelated generic statement here with enough length to pass.", "metadata": {"category": "rent"}}
        b1 = _boost(doc_with_section, ctx)
        b2 = _boost(doc_without, ctx)
        assert b1 > b2

    def test_noise_pattern_decreases_boost(self):
        ctx = understand_query("deposit refund", case_type="rent")
        noisy = {"text": "Table of contents short title this Act may be called the Test Act", "metadata": {}}
        b = _boost(noisy, ctx)
        assert b < 0.1  # heavily penalized

    def test_category_match_boosts(self):
        ctx = understand_query("deposit refund", case_type="rent")
        matching = {"text": "x" * 100, "metadata": {"category": "rent"}}
        non_matching = {"text": "x" * 100, "metadata": {"category": "consumer"}}
        assert _boost(matching, ctx) > _boost(non_matching, ctx)


class TestMMR:
    def test_mmr_returns_all_if_fewer_than_k(self):
        docs = [{"text": "a" * 50, "final_score": 0.9}]
        result = _mmr(docs, k=3)
        assert len(result) == 1

    def test_mmr_returns_k_items(self):
        docs = [
            {"text": "deposit refund landlord tenant section 108 case", "final_score": 0.9},
            {"text": "completely different topic about consumer goods refund", "final_score": 0.8},
            {"text": "another unrelated subject regarding employment wages", "final_score": 0.7},
            {"text": "deposit refund landlord tenant section 108 again", "final_score": 0.6},
        ]
        result = _mmr(docs, k=2)
        assert len(result) == 2
        assert result[0]["final_score"] == 0.9  # top doc always first


class TestSmartRetrieve:
    def test_smart_retrieve_returns_results(self, mock_retriever_rent):
        ctx = understand_query("deposit nahi mil raha hai", case_type="rent")
        results = smart_retrieve(ctx, final_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_smart_retrieve_empty_when_no_docs(self, mock_retriever_empty):
        ctx = understand_query("deposit nahi mil raha hai", case_type="rent")
        results = smart_retrieve(ctx, final_k=3)
        assert results == []


class TestBuildOptimizedContext:
    def test_returns_no_context_message_for_empty_docs(self):
        result = build_optimized_context([], max_chars=900)
        assert "No relevant legal context" in result

    def test_includes_law_name_header(self):
        result = build_optimized_context(SAMPLE_DOCS, max_chars=900)
        assert "Transfer of Property Act 1882" in result

    def test_respects_max_chars(self):
        result = build_optimized_context(SAMPLE_DOCS, max_chars=100)
        assert len(result) <= 120  # small overshoot allowed by trimming logic

    def test_skips_short_docs(self):
        short_docs = [{"text": "short", "metadata": {}}]
        result = build_optimized_context(short_docs, max_chars=900)
        assert "No relevant legal context" in result
