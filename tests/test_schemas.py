"""Tests for backend.api.schemas — Pydantic request/response validation."""
import pytest
from pydantic import ValidationError
from backend.api.schemas import (
    AskRequest, DeepAskRequest, ClassifyRequest, ChatRequest, ChatMessage,
    NoticeRequest, TimelineRequest,
)


class TestAskRequest:
    def test_valid_query_accepted(self):
        req = AskRequest(query="Landlord deposit nahi de raha hai")
        assert req.language == "en"

    def test_query_too_short_rejected(self):
        with pytest.raises(ValidationError):
            AskRequest(query="hi")

    def test_query_too_long_rejected(self):
        with pytest.raises(ValidationError):
            AskRequest(query="x" * 2001)

    def test_custom_language_accepted(self):
        req = AskRequest(query="Landlord deposit nahi de raha hai", language="hi")
        assert req.language == "hi"


class TestDeepAskRequest:
    def test_extra_context_optional(self):
        req = DeepAskRequest(query="Landlord deposit nahi de raha hai")
        assert req.extra_context == ""

    def test_extra_context_too_long_rejected(self):
        with pytest.raises(ValidationError):
            DeepAskRequest(query="Landlord deposit nahi de raha hai", extra_context="x" * 501)


class TestClassifyRequest:
    def test_minimum_length_enforced(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(query="hi")

    def test_valid_query(self):
        req = ClassifyRequest(query="Mera deposit wapas nahi mil raha")
        assert req.query


class TestChatRequest:
    def test_defaults_applied(self):
        req = ChatRequest(message="mera phone kho gaya")
        assert req.history == []
        assert req.case_type == "general"
        assert req.session_id == "default"

    def test_history_with_messages(self):
        req = ChatRequest(
            message="aur kya karu",
            history=[ChatMessage(role="user", content="phone chori ho gaya"),
                     ChatMessage(role="assistant", content="FIR file karo")],
        )
        assert len(req.history) == 2
        assert req.history[0].role == "user"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_message_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 1001)


class TestNoticeRequest:
    def test_valid_notice_type_accepted(self):
        req = NoticeRequest(
            notice_type="deposit_refund",
            sender_name="A", sender_address="Addr A",
            recipient_name="B", recipient_address="Addr B",
            facts="Landlord has not refunded deposit for three months despite requests",
            relief="Refund deposit", law="TPA Section 108",
        )
        assert req.notice_type == "deposit_refund"
        assert req.generate_pdf is False

    def test_invalid_notice_type_rejected(self):
        with pytest.raises(ValidationError):
            NoticeRequest(
                notice_type="not_a_valid_type",
                sender_name="A", sender_address="Addr A",
                recipient_name="B", recipient_address="Addr B",
                facts="Landlord has not refunded deposit for three months despite requests",
                relief="Refund deposit", law="TPA Section 108",
            )

    def test_facts_too_short_rejected(self):
        with pytest.raises(ValidationError):
            NoticeRequest(
                notice_type="general",
                sender_name="A", sender_address="Addr A",
                recipient_name="B", recipient_address="Addr B",
                facts="too short",
                relief="Refund deposit", law="TPA Section 108",
            )


class TestTimelineRequest:
    def test_valid_request(self):
        req = TimelineRequest(case_type="rent", facts="Deposit issue", outcome="recovery suit")
        assert req.case_type == "rent"
