"""main.py — Production FastAPI v3. Run: uvicorn main:app --reload --reload-dir backend --port 8000"""
import os, time, uuid, logging
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.api.routes import router
from backend.core.rate_limit import is_allowed, stats as rate_limit_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger   = logging.getLogger(__name__)
_metrics = defaultdict(int)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Startup] Loading FAISS + model...")
    from backend.rag.loader import initialize
    initialize()
    logger.info("[Startup] ✅ Ready")
    _metrics["startup_count"] += 1
    yield
    logger.info("[Shutdown] Done.")

app = FastAPI(title="AI Legal Advisor — India", description="Production RAG legal guidance v3.",
              version="3.0.0", docs_url="/docs", redoc_url="/redoc", lifespan=lifespan)

allowed_origins = [origin.strip() for origin in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8501"
).split(",") if origin.strip()]

app.add_middleware(CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True, allow_methods=["GET","POST"], allow_headers=["Content-Type"])

MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(1_000_000)))  # 1MB — plenty for this API's JSON payloads

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    # Pydantic's max_length constraints only kick in AFTER the body is read
    # and parsed. A client can still declare (and send) a multi-GB body
    # before that point, exhausting memory/bandwidth. Reject oversized
    # bodies up front based on Content-Length. Not airtight (a client could
    # omit Content-Length and stream chunked), but it stops the common case
    # cheaply, for negligible cost on every other request.
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"error": "Request body too large."})
    return await call_next(request)

@app.middleware("http")
async def mw(request: Request, call_next):
    rid  = str(uuid.uuid4())[:8]
    request.state.request_id = rid

    # Rate limit only the LLM-backed API surface (/api/v2/*) — cheap to skip
    # for /health, /docs, /metrics, so those keep working during a burst.
    # This protects against both abusive traffic and runaway Groq API cost.
    if request.url.path.startswith("/api/v2/"):
        forwarded = request.headers.get("x-forwarded-for", "")
        client_ip = (
            forwarded.split(",")[0].strip() if forwarded
            else (request.client.host if request.client else "unknown")
        )
        allowed, retry_after = is_allowed(client_ip)
        if not allowed:
            _metrics["rate_limited_requests"] += 1
            logger.warning(f"[RateLimit] {client_ip} blocked on {request.url.path} (retry_after={retry_after}s)")
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Please slow down.", "request_id": rid},
                headers={"Retry-After": str(retry_after), "X-Request-ID": rid},
            )

    t0   = time.perf_counter()
    resp = await call_next(request)
    ms   = (time.perf_counter()-t0)*1000
    resp.headers["X-Request-ID"]           = rid
    resp.headers["X-Response-Time"]        = f"{ms:.0f}ms"
    # Baseline security headers — safe defaults for a JSON API that is not
    # meant to be framed/embedded and should never be MIME-sniffed.
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"]        = "DENY"
    resp.headers["Referrer-Policy"]        = "no-referrer"
    logger.info(f"[{request.method}] {request.url.path} {resp.status_code} {ms:.0f}ms")
    _metrics["total_requests"] += 1
    _metrics[f"status_{resp.status_code}"] += 1
    if ms > 1500: _metrics["slow_requests"] += 1
    return resp

app.include_router(router, prefix="/api/v2")

@app.get("/")
def root():
    return {"service":"AI Legal Advisor India","version":"3.0.0","docs":"/docs","api":"/api/v2"}

@app.get("/health")
def health():
    try:
        from backend.rag.loader import get_index
        from backend.core.cache import cache_stats
        from backend.v2.memory import get_memory
        return {"status":"healthy","faiss_vectors":get_index().ntotal,
                "cache":cache_stats(),"memory":get_memory().stats(),
                "rate_limit":rate_limit_stats(),"metrics":dict(_metrics)}
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"status":"unhealthy","error":str(e)})

@app.get("/metrics")
def metrics(): return dict(_metrics)
