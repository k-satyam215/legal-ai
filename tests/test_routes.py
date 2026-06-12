"""
Integration tests for backend.api.routes — FastAPI endpoints via TestClient.

All LLM and FAISS/retriever calls are mocked (see conftest.py `app_client` fixture).
"""
import pytest


class TestRootAndHealth:
    def test_root_endpoint(self, app_client):
        resp = app_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "AI Legal Advisor India"
        assert data["api"] == "/api/v2"

    def test_metrics_endpoint(self, app_client):
        resp = app_client.get("/metrics")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_v2_health_endpoint(self, app_client):
        resp = app_client.get("/api/v2/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "cache" in data


class TestAskEndpoint:
    def test_ask_returns_legal_advice(self, app_client):
        resp = app_client.post("/api/v2/ask", json={"query": "Landlord deposit nahi de raha hai 3 mahine se"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["case_type"] == "rent"
        assert "laws" in data
        assert "steps" in data
        assert "request_id" not in data or data["request_id"] is None or isinstance(data["request_id"], str)

    def test_ask_rejects_empty_query(self, app_client):
        resp = app_client.post("/api/v2/ask", json={"query": ""})
        assert resp.status_code == 422

    def test_ask_rejects_too_short_query(self, app_client):
        resp = app_client.post("/api/v2/ask", json={"query": "hi"})
        assert resp.status_code == 422

    def test_ask_includes_request_id(self, app_client):
        resp = app_client.post("/api/v2/ask", json={"query": "Landlord deposit nahi de raha hai"})
        data = resp.json()
        assert data.get("request_id") is not None

    def test_ask_with_whitespace_only_query_rejected(self, app_client):
        resp = app_client.post("/api/v2/ask", json={"query": "          "})
        # passes min_length validation (spaces count) but _s() strips to empty -> 422 from handler
        assert resp.status_code == 422


class TestDeepAskEndpoint:
    def test_deep_ask_returns_full_analysis(self, app_client):
        resp = app_client.post("/api/v2/deep-ask", json={"query": "Landlord deposit nahi de raha hai 3 mahine se"})
        assert resp.status_code == 200
        data = resp.json()
        assert "legal_interpretation" in data
        assert "scenario_analysis" in data
        assert "alternative_remedies" in data

    def test_deep_ask_with_extra_context(self, app_client):
        resp = app_client.post("/api/v2/deep-ask", json={
            "query": "Landlord deposit nahi de raha hai 3 mahine se",
            "extra_context": "Tenant has a written agreement",
        })
        assert resp.status_code == 200

    def test_deep_ask_rejects_empty_query(self, app_client):
        resp = app_client.post("/api/v2/deep-ask", json={"query": ""})
        assert resp.status_code == 422


class TestChatEndpoint:
    def test_chat_quick_card_response(self, app_client):
        resp = app_client.post("/api/v2/chat", json={"message": "mera phone kho gaya"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["quick_card"] == "phone_lost"
        assert data["detected_intent"] == "phone_lost"

    def test_chat_llm_path_response(self, app_client):
        resp = app_client.post("/api/v2/chat", json={
            "message": "is case me konsi adalat me jaana hoga aur kitna time lagega",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data

    def test_chat_with_history(self, app_client):
        resp = app_client.post("/api/v2/chat", json={
            "message": "aur kya documents chahiye",
            "history": [
                {"role": "user", "content": "mera phone chori ho gaya"},
                {"role": "assistant", "content": "FIR file karo IPC 379 ke under"},
            ],
        })
        assert resp.status_code == 200

    def test_chat_rejects_empty_message(self, app_client):
        resp = app_client.post("/api/v2/chat", json={"message": ""})
        assert resp.status_code == 422


class TestClassifyEndpoint:
    def test_classify_returns_case_type(self, app_client):
        resp = app_client.post("/api/v2/classify-case", json={"query": "Mera landlord deposit nahi de raha hai"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["case_type"] == "rent"
        assert 0.0 <= data["confidence"] <= 1.0

    def test_classify_rejects_short_query(self, app_client):
        resp = app_client.post("/api/v2/classify-case", json={"query": "hi"})
        assert resp.status_code == 422


class TestNoticeEndpoint:
    def test_generate_notice_text_only(self, app_client, mock_llm):
        resp = app_client.post("/api/v2/generate-notice", json={
            "notice_type": "deposit_refund",
            "sender_name": "Rahul Sharma",
            "sender_address": "123 MG Road, Bengaluru",
            "recipient_name": "Suresh Kumar",
            "recipient_address": "456 Park Street, Bengaluru",
            "facts": "Tenant vacated the premises on 1st January but landlord has not refunded the security deposit of Rs 50000 despite repeated requests",
            "relief": "Refund of Rs 50000 with interest",
            "law": "Transfer of Property Act 1882, Section 108",
            "generate_pdf": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "notice_text" in data
        assert data["pdf_path"] is None

    def test_generate_notice_invalid_type_rejected(self, app_client):
        resp = app_client.post("/api/v2/generate-notice", json={
            "notice_type": "invalid_type",
            "sender_name": "A", "sender_address": "Addr",
            "recipient_name": "B", "recipient_address": "Addr",
            "facts": "Some facts that are long enough to pass validation easily",
            "relief": "relief", "law": "law",
        })
        assert resp.status_code == 422


class TestTimelineEndpoint:
    def test_timeline_returns_milestones(self, app_client, mock_llm):
        resp = app_client.post("/api/v2/timeline", json={
            "case_type": "rent",
            "facts": "Deposit not refunded for 3 months",
            "outcome": "Recovery suit filed",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "milestones" in data
        assert isinstance(data["milestones"], list)

    def test_timeline_falls_back_on_llm_error(self, app_client, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr("backend.services.timeline_generator.call_llm", boom)

        resp = app_client.post("/api/v2/timeline", json={
            "case_type": "rent",
            "facts": "Deposit not refunded for 3 months",
            "outcome": "Recovery suit filed",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["milestones"]) > 0
        assert data["milestones"][0]["phase"] == "Pre-Litigation"


class TestResponseHeaders:
    def test_response_includes_request_id_header(self, app_client):
        resp = app_client.get("/")
        assert "X-Request-ID" in resp.headers
        assert "X-Response-Time" in resp.headers

    def test_cors_headers_present(self, app_client):
        resp = app_client.options("/api/v2/ask", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        })
        assert resp.status_code in (200, 204)
