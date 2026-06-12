"""Tests for backend.core.cache — TTL cache."""
import time
import pytest
from backend.core.cache import cache_get, cache_set, cache_stats, cache_clear, _key, MAX_CACHE_SIZE


class TestKeyGeneration:
    def test_key_is_deterministic(self):
        assert _key("query", "rent") == _key("query", "rent")

    def test_key_case_insensitive_on_query(self):
        assert _key("Query Text", "rent") == _key("query text", "rent")

    def test_different_case_types_different_keys(self):
        assert _key("query", "rent") != _key("query", "consumer")


class TestCacheGetSet:
    def test_set_then_get_returns_data(self):
        cache_clear()
        cache_set("test query", {"answer": "42"}, case_type="general", ttl=60)
        result = cache_get("test query", "general")
        assert result == {"answer": "42"}

    def test_get_missing_key_returns_none(self):
        cache_clear()
        assert cache_get("nonexistent query", "general") is None

    def test_expired_entry_returns_none(self):
        cache_clear()
        cache_set("expiring query", {"x": 1}, case_type="general", ttl=-1)
        assert cache_get("expiring query", "general") is None

    def test_different_case_type_different_cache_entry(self):
        cache_clear()
        cache_set("same query", {"type": "rent"}, case_type="rent", ttl=60)
        assert cache_get("same query", "consumer") is None
        assert cache_get("same query", "rent") == {"type": "rent"}


class TestCacheStats:
    def test_stats_reflects_active_keys(self):
        cache_clear()
        cache_set("q1", {"a": 1}, case_type="general", ttl=60)
        cache_set("q2", {"a": 2}, case_type="general", ttl=60)
        stats = cache_stats()
        assert stats["total_keys"] == 2
        assert stats["active_keys"] == 2

    def test_stats_on_empty_cache(self):
        cache_clear()
        stats = cache_stats()
        assert stats["total_keys"] == 0
        assert stats["active_keys"] == 0


class TestCacheEviction:
    def test_evicts_oldest_when_full(self):
        cache_clear()
        for i in range(MAX_CACHE_SIZE + 5):
            cache_set(f"query {i}", {"i": i}, case_type="general", ttl=3600)
        stats = cache_stats()
        assert stats["total_keys"] <= MAX_CACHE_SIZE


class TestCacheClear:
    def test_clear_empties_cache(self):
        cache_set("q", {"a": 1}, case_type="general", ttl=60)
        cache_clear()
        assert cache_stats()["total_keys"] == 0
