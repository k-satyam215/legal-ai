---
title: AI Legal Advisor India
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# AI Legal Advisor — India v2 (Production)

![CI](https://github.com/k-satyam215/legal-ai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Stack: FastAPI + FAISS + LLaMA-3.3-70b + Streamlit

A production-grade RAG legal advisor for Indian law — strict grounded citations (zero hallucination), Hindi/Hinglish-aware classification, multi-turn conversational memory, legal notice & timeline generation, and a dark-themed Streamlit UI.

---

## Folder Structure

```
legal-ai/
├── backend/
│   ├── api/
│   │   ├── routes.py          ← All endpoints incl. /chat
│   │   └── schemas.py
│   ├── core/
│   │   ├── cache.py           ← TTL cache (1hr, Redis-upgradeable)
│   │   ├── config.py
│   │   ├── llm.py
│   │   └── prompts.py         ← Strict legal prompts
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── loader.py
│   │   └── retriever.py
│   ├── services/
│   │   ├── notice_generator.py
│   │   └── timeline_generator.py
│   └── v2/                    ← All production logic
│       ├── chat_engine.py     ← NEW: conversational chat
│       ├── classifier_v2.py
│       ├── legal_advisor_v2.py
│       ├── query_understanding.py
│       ├── router_v2.py
│       └── smart_retriever.py
├── data_pipeline/
│   ├── cleaner/clean_text.py
│   ├── chunker/chunking.py
│   ├── embeddings/embedder.py
│   ├── evaluation/
│   │   ├── eval_v2.py
│   │   └── test_cases.json
│   ├── retriever/
│   │   ├── build_index.py
│   │   └── query_index.py
│   ├── scraper/
│   │   ├── indiancode_scraper.py
│   │   └── indiankanoon_scraper.py
│   ├── structured_data/case_templates.json
│   └── pipeline.py
├── frontend/
│   └── app.py                 ← Dark UI with Chat tab
├── tests/                     ← 100+ pytest tests, mocked LLM/FAISS
│   ├── conftest.py
│   ├── test_routes.py
│   ├── test_classifier.py
│   ├── test_query_understanding.py
│   ├── test_smart_retriever.py
│   ├── test_legal_advisor.py
│   ├── test_chat_engine.py
│   ├── test_memory.py
│   ├── test_cache.py
│   └── test_schemas.py
├── .github/workflows/ci.yml   ← Lint + test + Docker build
├── Dockerfile                 ← Backend image
├── Dockerfile.frontend        ← Streamlit image
├── docker-compose.yml
├── .env.example
├── main.py
└── requirements.txt
```

---

## Setup

### Option A — Local (Python)

```bash
# 1. Add your data files:
backend/rag/faiss_index.index
backend/rag/faiss_index_metadata.json
backend/rag/faiss_index_offsets.npy
backend/rag/bm25_index.pkl
backend/data/   ← your PDFs

# 2. Setup env
cp .env.example .env
# set GROQ_API_KEY in .env

# 3. Install
pip install -r requirements.txt

# 4. Run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 5. Run frontend (new terminal)
cd frontend && streamlit run app.py
```

### Option B — Docker

```bash
cp .env.example .env
# set GROQ_API_KEY in .env

docker compose up --build
# Backend  → http://localhost:8000
# Frontend → http://localhost:8501
```

---

## Testing

100+ tests covering classification, RAG retrieval/re-ranking, legal advisor validation & fallbacks, chat engine, memory, cache, schemas, and full API integration — all LLM and FAISS calls are mocked, so no API key or index is required to run the suite.

```bash
pip install -r requirements.txt
pytest -v

# with coverage
pytest --cov=backend --cov-report=term-missing
```

CI runs lint (`ruff`) + the full test suite + Docker image builds on every push via GitHub Actions (`.github/workflows/ci.yml`).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /api/v2/ask | Deep legal Q&A (strict JSON) |
| POST | /api/v2/chat | **NEW** Conversational chat |
| POST | /api/v2/classify-case | Classify query |
| POST | /api/v2/generate-notice | Legal notice |
| POST | /api/v2/timeline | Legal timeline |
| GET  | /health | FAISS + cache status |
| GET  | /docs | Swagger UI |

### Chat example
```bash
curl -X POST http://localhost:8000/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "mera phone kho gaya kya karun", "history": []}'
```

### Ask example
```bash
curl -X POST http://localhost:8000/api/v2/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "landlord deposit 50000 nahi de raha 3 mahine ho gaye"}'
```

---

## What's New in v2

| Feature | Detail |
|---|---|
| **Legal Chat** | Multi-turn conversational assistant. Quick reference cards for common issues. No RAG for instant replies. |
| **Strict Prompts** | max_tokens=220, exact sections mandatory, no generic advice |
| **Intent Expansion** | "phone lost" → ["IPC 379","CrPC 154","CEIR portal"] |
| **Legal Section Preference** | Chunks with section numbers boosted over raw judgments |
| **Grounded Citations** | From retrieved docs only — zero hallucination |
| **Speed** | Target 800–1200ms. Cache hit <50ms. |
| **Dark UI** | Professional dark theme, chat bubbles, metrics |

---

## Security

| Measure | Detail |
|---|---|
| **Rate limiting** | In-memory sliding window on all `/api/v2/*` routes (`RATE_LIMIT_MAX_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS`, default 30 req/60s per client IP). Returns `429` with `Retry-After`. Protects against abuse and runaway Groq API cost. |
| **Request body size cap** | Declared bodies over `MAX_BODY_BYTES` (default 1MB) are rejected with `413` before Pydantic parses them, so an oversized payload can't be used for memory/bandwidth exhaustion. |
| **Per-session isolation** | `/api/v2/chat` mints a fresh random `session_id` (`secrets.token_hex`) server-side when the client doesn't supply one, and echoes it back in the response. Previously defaulted to the literal string `"default"`, so any two clients that omitted it shared one conversation-memory bucket — fixed. |
| **Security headers** | Every response sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`. |
| **Strict CORS** | Origins restricted via `ALLOWED_ORIGINS`; methods limited to `GET`/`POST`; headers limited to `Content-Type`. |
| **Input validation** | Pydantic length limits on every request field; control characters stripped (`_s()` in `routes.py`); `notice_type` restricted to an allow-list. |
| **PDF-injection guard** | Notice text is HTML/XML-escaped before being handed to ReportLab's `Paragraph`, since it embeds LLM/user-supplied text and interprets a subset of markup. |
| **No secret leakage** | Unhandled exceptions are logged server-side only; the client gets a generic message + `request_id` for support lookup (`_internal_error()` in `routes.py`). |
| **XSS-safe frontend** | All dynamic text is HTML-escaped (`h()` in `frontend/app.py`) before being rendered via `unsafe_allow_html`. |
| **Non-root containers** | All three Dockerfiles (`Dockerfile`, `Dockerfile.backend`, `Dockerfile.frontend`) create and switch to an unprivileged `appuser` before `CMD`, limiting blast radius if the app is ever compromised. |

Rate limiting is in-memory and per-process by design (same graceful-fallback philosophy as the Redis cache) — for a multi-instance deployment, back it with Redis the same way `backend/core/cache.py` does.
