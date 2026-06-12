"""main.py — Production FastAPI v3. Run: uvicorn main:app --reload --reload-dir backend --port 8000"""
import os, time, uuid, logging
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.api.routes import router

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

app.add_middleware(CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS","*").split(","),
    allow_credentials=True, allow_methods=["GET","POST"], allow_headers=["*"])

@app.middleware("http")
async def mw(request: Request, call_next):
    rid  = str(uuid.uuid4())[:8]
    request.state.request_id = rid
    t0   = time.perf_counter()
    resp = await call_next(request)
    ms   = (time.perf_counter()-t0)*1000
    resp.headers["X-Request-ID"]    = rid
    resp.headers["X-Response-Time"] = f"{ms:.0f}ms"
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
                "cache":cache_stats(),"memory":get_memory().stats(),"metrics":dict(_metrics)}
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={"status":"unhealthy","error":str(e)})

@app.get("/metrics")
def metrics(): return dict(_metrics)
