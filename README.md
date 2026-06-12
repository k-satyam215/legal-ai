# AI Legal Advisor — India v2 (Production)

![CI](https://github.com/<your-username>/legal-ai/actions/workflows/ci.yml/badge.svg)
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
