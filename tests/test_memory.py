"""Tests for backend.v2.memory — session memory, topic detection, follow-up logic."""
import time
import pytest
from backend.v2.memory import (
    MemoryStore, CaseState, _detect_topic, _topics_differ,
    format_history_for_llm, is_followup, make_session_id,
)


class TestDetectTopic:
    def test_detects_rent_topic(self):
        assert _detect_topic("landlord deposit nahi de raha") == "rent"

    def test_detects_consumer_topic(self):
        assert _detect_topic("amazon se defective product mila refund nahi") == "consumer"

    def test_detects_criminal_topic(self):
        assert _detect_topic("fir file karna hai police chori") == "criminal"

    def test_unmatched_text_defaults_general(self):
        assert _detect_topic("xyz abc nonsense") == "general"


class TestTopicsDiffer:
    def test_same_topic_does_not_differ(self):
        assert _topics_differ("rent", "rent") is False

    def test_general_never_differs(self):
        assert _topics_differ("general", "rent") is False
        assert _topics_differ("rent", "general") is False

    def test_different_specific_topics_differ(self):
        assert _topics_differ("rent", "criminal") is True


class TestCaseState:
    def test_empty_state_returns_empty_string(self):
        cs = CaseState()
        assert cs.to_context_string() == ""

    def test_populated_state_includes_issue(self):
        cs = CaseState(issue="Deposit not refunded", case_type="rent")
        ctx = cs.to_context_string()
        assert "Deposit not refunded" in ctx
        assert "rent" in ctx

    def test_includes_facts_and_laws(self):
        cs = CaseState(facts=["Paid 50000 deposit"], laws_discussed=["TPA Section 108"])
        ctx = cs.to_context_string()
        assert "50000" in ctx
        assert "TPA Section 108" in ctx


class TestMemoryStore:
    def test_get_or_create_creates_new_session(self):
        store = MemoryStore()
        session = store.get_or_create("abc123")
        assert session.session_id == "abc123"
        assert session.turns == []

    def test_add_turn_appends(self):
        store = MemoryStore()
        store.add_turn("s1", "user", "deposit nahi de raha landlord")
        session = store.get_or_create("s1")
        assert len(session.turns) == 1
        assert session.turns[0].role == "user"

    def test_turn_limit_enforced(self):
        store = MemoryStore()
        for i in range(15):
            store.add_turn("s1", "user", f"message {i} about rent deposit")
        session = store.get_or_create("s1")
        assert len(session.turns) <= 8  # MAX_TURNS

    def test_topic_change_resets_case_state(self):
        store = MemoryStore()
        store.add_turn("s1", "user", "mera landlord deposit nahi de raha hai abhi")
        store.update_case_state("s1", issue="Deposit issue")
        store.add_turn("s1", "user", "mera phone chori ho gaya hai FIR file karna hai police")
        session = store.get_or_create("s1")
        # Topic changed from rent -> criminal, case state should reset
        assert session.case_state.issue == ""

    def test_clear_removes_session(self):
        store = MemoryStore()
        store.add_turn("s1", "user", "test message")
        store.clear("s1")
        session = store.get_or_create("s1")
        assert session.turns == []  # new session created fresh

    def test_stats_returns_counts(self):
        store = MemoryStore()
        store.add_turn("s1", "user", "hello")
        stats = store.stats()
        assert stats["total"] >= 1
        assert "active" in stats

    def test_update_case_state_bounds_lists(self):
        store = MemoryStore()
        store.get_or_create("s1")
        for i in range(10):
            store.update_case_state("s1", facts=[f"fact {i}"])
        session = store.get_or_create("s1")
        assert len(session.case_state.facts) <= 5


class TestFormatHistoryForLLM:
    def test_empty_history_returns_none_string(self):
        assert format_history_for_llm([]) == "None"

    def test_formats_user_and_assistant_turns(self):
        history = [
            {"role": "user", "content": "deposit issue"},
            {"role": "assistant", "content": "file FIR"},
        ]
        result = format_history_for_llm(history)
        assert "User:" in result
        assert "Assistant:" in result

    def test_truncates_long_content(self):
        history = [{"role": "user", "content": "x" * 500}]
        result = format_history_for_llm(history)
        assert "..." in result

    def test_respects_max_turns(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = format_history_for_llm(history, max_turns=2)
        assert result.count("User:") == 2


class TestIsFollowup:
    def test_empty_history_not_followup(self):
        assert is_followup("kya documents chahiye", []) is False

    def test_short_message_with_signal_is_followup(self):
        history = [{"role": "user", "content": "deposit issue"}]
        assert is_followup("aur kya documents chahiye", history) is True

    def test_long_message_without_signal_not_followup(self):
        history = [{"role": "user", "content": "deposit issue"}]
        msg = "This is a completely new and unrelated long legal question about something else entirely"
        assert is_followup(msg, history) is False


class TestMakeSessionId:
    def test_returns_string_of_expected_length(self):
        sid = make_session_id("1.2.3.4", "test-agent")
        assert isinstance(sid, str)
        assert len(sid) == 12

    def test_same_inputs_same_window_produce_same_id(self):
        sid1 = make_session_id("1.2.3.4", "agent")
        sid2 = make_session_id("1.2.3.4", "agent")
        assert sid1 == sid2
