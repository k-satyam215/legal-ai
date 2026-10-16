---
title: AI Legal Advisor India
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# AI Legal Advisor — India

![CI](https://github.com/k-satyam215/legal-ai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Stack:** FastAPI · FAISS + BM25 hybrid retrieval · Groq (`openai/gpt-oss-120b`) · LLM Guard · Redis (optional) · Streamlit

A security-conscious RAG legal assistant for Indian law: citation-grounded answers, Hindi/Hinglish-aware intent handling, multi-turn chat memory, legal notice and timeline generation, and layered prompt-injection defenses.

> Informational tool only — not a substitute for advice from a qualified advocate.

---

## Architecture

```
Client (Streamlit / curl)
        │
        ▼
FastAPI  ── middleware: body-size cap → rate limit → security headers
        │
        ├─ sanitize + ML injection scan (LLM Guard) on /ask, /deep-ask, /chat, /generate-notice
        │
        ▼
classifier_v2 ─► smart_retriever (FAISS dense + BM25 sparse, section-preferring re-rank)
        │
        ▼
legal_advisor_v2 / chat_engine ─► Groq LLM (tagged prompts, strict JSON) ─► response
        │
        └─ Redis TTL cache (falls back to in-memory)
```

## Folder structure

```
legal-ai/
├── main.py                      FastAPI app, middleware (body cap, rate limit, headers, CORS)
├── backend/
│   ├── api/                     routes.py, schemas.py
│   ├── core/
│   │   ├── cache.py             Redis TTL cache with in-memory fallback
│   │   ├── config.py
│   │   ├── injection_scanner.py LLM Guard (DeBERTa) prompt-injection scanner, fail-open
│   │   ├── llm.py               Groq client with retries
│   │   ├── prompts.py           Strict, tag-delimited prompts
│   │   ├── rate_limit.py        In-memory sliding-window limiter
│   │   └── security.py          Sanitization, tag neutralization, regex heuristic
│   ├── rag/                     chunking, loader, retriever (FAISS + BM25)
│   ├── services/                notice_generator.py, timeline_generator.py
│   └── v2/                      chat_engine, classifier, legal_advisor, memory,
│                                query_understanding, router, smart_retriever
├── data_pipeline/               scrapers, cleaning, chunking, embeddings, index build, eval
├── frontend/app.py              Streamlit UI
├── tests/                       pytest suite (LLM, FAISS and the ML scanner are mocked)
├── .github/workflows/           ci.yml (lint, test, Docker build), hf-deploy.yml
├── Dockerfile · Dockerfile.backend · Dockerfile.frontend · docker-compose.yml
└── requirements.txt
```

## Setup

### Local

```bash
# 1. Data files (built by data_pipeline/ or downloaded at deploy time):
#    backend/rag/faiss_index.index, faiss_index_metadata.json,
#    faiss_index_offsets.npy, bm25_index.pkl

cp .env.example .env          # set GROQ_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
cd frontend && streamlit run app.py
```

### Docker

```bash
cp .env.example .env          # set GROQ_API_KEY
docker compose up --build
# Backend  → http://localhost:8000      Frontend → http://localhost:8501
```

## Testing

```bash
pip install -r requirements.txt
pytest -q
pytest --cov=backend --cov-report=term-missing
```

All LLM, FAISS and LLM Guard model calls are mocked, so the suite needs no API key, index, or model download. `tests/conftest.py` sets `ENABLE_ML_INJECTION_SCAN=false` so the ~700MB model is never loaded in tests; `tests/test_injection_scanner.py` exercises the scanner and the API gate with a stub model.

CI (GitHub Actions) runs ruff, the test suite with coverage, and Docker image builds. Lint is currently advisory (it does not fail the build).

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/v2/ask` | Legal Q&A (strict JSON, cached) |
| POST | `/api/v2/deep-ask` | Deep analysis with scenarios and strategy |
| POST | `/api/v2/chat` | Multi-turn chat with quick-reference cards and memory |
| POST | `/api/v2/classify-case` | Classify a query into a case type |
| POST | `/api/v2/generate-notice` | Legal notice text (optional PDF) |
| POST | `/api/v2/timeline` | Case timeline milestones |
| GET | `/health` | FAISS, cache, memory and rate-limit status |
| GET | `/metrics` | Request counters |
| GET | `/docs` | Swagger UI |

```bash
curl -X POST http://localhost:8000/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "mera phone kho gaya kya karun", "history": []}'
```

## Features

| Feature | Detail |
|---|---|
| Hybrid retrieval | FAISS dense + BM25 sparse, with a boost for chunks that carry statute section numbers |
| Intent expansion | "phone lost" → IPC 379, CrPC 154, CEIR portal |
| Grounded answers | Citations come only from retrieved documents; strict JSON output |
| Chat | Multi-turn memory, follow-up detection, instant quick-reference cards for common issues |
| Caching | Redis TTL cache (1 hr) with in-memory fallback; cached repeat queries return in tens of ms |
| Languages | English, Hindi and Hinglish queries |

## Security

| Layer | Detail |
|---|---|
| **Prompt-injection scanning** | LLM Guard's DeBERTa `PromptInjection` classifier runs on `/ask`, `/deep-ask`, `/chat` and `/generate-notice` input before any LLM call; hits return `400` (or a refusal reply in chat). Layered with a regex heuristic and tag-delimited prompts (`<user_query>`, `<retrieved_context>`) whose angle brackets are neutralized so input cannot forge a boundary. |
| **Fail-open by design** | If the model cannot load (memory limits, download failure) or `ENABLE_ML_INJECTION_SCAN=false`, the scanner reports "clean" and the regex + tag-fencing layers still apply, so a constrained deployment never goes down because of this optional layer. |
| **Rate limiting** | In-memory sliding window on `/api/v2/*` (default 30 req / 60 s per client IP), `429` + `Retry-After`. Tracked-key table is capped. |
| **Request size cap** | Bodies declaring more than `MAX_BODY_BYTES` (1 MB) get `413` before parsing. |
| **Per-session isolation** | `/chat` mints a random `session_id` server-side when the client omits one. |
| **Input validation** | Pydantic length limits, control characters stripped, `notice_type` allow-list. |
| **Security headers** | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`. |
| **Strict CORS** | Origins from `ALLOWED_ORIGINS`; `GET`/`POST` only; `Content-Type` header only. |
| **PDF-injection guard** | Notice text is HTML/XML-escaped before ReportLab. |
| **No secret leakage** | Exceptions are logged server-side; clients get a generic message plus `request_id`. |
| **XSS-safe frontend** | Dynamic text is escaped before `unsafe_allow_html`. |
| **Non-root containers** | All Dockerfiles drop to an unprivileged user. |

### Known limitations

- Rate limiting is per-process and in-memory. Behind several replicas each worker limits independently; back it with Redis for a multi-instance deployment.
- The limiter keys on the first `X-Forwarded-For` entry, which a client can set. Deploy behind a trusted proxy that overwrites the header, or switch to the last entry added by your proxy.
- ML injection detection is probabilistic; it reduces risk but is not a guarantee. It is one layer, not the only one.
