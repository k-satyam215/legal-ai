"""API Routes — /api/v2"""
import re, logging, secrets, tempfile, os
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from backend.v2.router_v2 import route_query_v2, route_deep_analysis
from backend.v2.classifier_v2 import classify_query_v2
from backend.v2.chat_engine import chat_response
from backend.core.injection_scanner import scan as scan_injection
from backend.services.notice_generator import generate_notice_text, generate_notice_pdf
from backend.services.timeline_generator import generate_timeline
from backend.api.schemas import (
    AskRequest, LegalAdviceResponse, DeepAskRequest, DeepAnalysisResponse,
    ClassifyRequest, ClassifyResponse, ChatRequest, ChatResponse,
    NoticeRequest, NoticeResponse, TimelineRequest, TimelineResponse,
)
logger = logging.getLogger(__name__)
router = APIRouter()


def _reject_if_injection(*texts) -> None:
    """Run the ML prompt-injection scanner on every free-text field that will
    reach an LLM prompt. Raises 400 on a hit. The scanner fails open (returns
    clean) if the model can't load, so this never takes the API down."""
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            continue
        is_inj, score = scan_injection(t)
        if is_inj:
            logger.warning("[Routes] Blocked suspected prompt injection (score=%.2f)", score)
            raise HTTPException(
                status_code=400,
                detail="Request looks like a prompt-injection attempt. Please describe your legal issue in plain language.",
            )

def _s(text: str) -> str:
    return " ".join(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]","",text).split()).strip()


def _internal_error(endpoint: str, request: Request | None = None) -> HTTPException:
    """Log diagnostics server-side without exposing implementation details."""
    request_id = getattr(getattr(request, "state", None), "request_id", "?")
    logger.exception("Unhandled error in %s [request_id=%s]", endpoint, request_id)
    return HTTPException(
        status_code=500,
        detail={"error": "Unable to process the request right now.", "request_id": request_id},
    )

@router.post("/ask", response_model=LegalAdviceResponse, tags=["Legal Advisor"])
async def ask(request: AskRequest, req: Request):
    try:
        q = _s(request.query)
        if not q: raise HTTPException(status_code=422, detail="Empty query")
        _reject_if_injection(q)
        return route_query_v2(q, language=request.language or "en")
    except HTTPException: raise
    except Exception:
        raise _internal_error("/ask", req)

@router.post("/deep-ask", response_model=DeepAnalysisResponse, tags=["Legal Advisor"])
async def deep_ask(request: DeepAskRequest, req: Request):
    try:
        q = _s(request.query)
        if not q: raise HTTPException(status_code=422, detail="Empty query")
        _reject_if_injection(q, request.extra_context or "")
        return route_deep_analysis(q, extra_context=request.extra_context or "", language=request.language or "en")
    except HTTPException: raise
    except Exception:
        raise _internal_error("/deep-ask", req)

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, req: Request):
    try:
        msg     = _s(request.message)
        history = [{"role":m.role,"content":m.content} for m in request.history]
        # If the client didn't supply a session_id, mint a fresh random one
        # instead of falling back to a shared value — see the comment on
        # ChatRequest.session_id. A client-supplied id is still sanitized
        # (control chars stripped) since it's used as a MemoryStore dict key.
        session_id = _s(request.session_id) if request.session_id else secrets.token_hex(8)
        if not session_id:
            session_id = secrets.token_hex(8)
        result  = chat_response(msg, history, case_type=request.case_type or "general", session_id=session_id)
        result["session_id"] = session_id
        return ChatResponse(**result)
    except Exception:
        raise _internal_error("/chat", req)

@router.post("/classify-case", response_model=ClassifyResponse, tags=["Classification"])
async def classify_case(request: ClassifyRequest, req: Request):
    try: return classify_query_v2(_s(request.query))
    except Exception: raise _internal_error("/classify-case", req)

@router.post("/generate-notice", response_model=NoticeResponse, tags=["Notice"])
async def generate_notice_api(request: NoticeRequest, background_tasks: BackgroundTasks, req: Request):
    _reject_if_injection(request.facts, request.relief)
    try:
        text = generate_notice_text(
            notice_type=request.notice_type, sender_name=request.sender_name,
            sender_address=request.sender_address, recipient_name=request.recipient_name,
            recipient_address=request.recipient_address, facts=request.facts,
            relief=request.relief, law=request.law,
        )
        pdf_path = None
        if request.generate_pdf:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            pdf_path = generate_notice_pdf(text, request.sender_name, output_path=tmp.name)
            background_tasks.add_task(os.unlink, tmp.name)
        return NoticeResponse(notice_text=text, pdf_path=pdf_path)
    except Exception:
        raise _internal_error("/generate-notice", req)

@router.post("/timeline", response_model=TimelineResponse, tags=["Timeline"])
async def get_timeline(request: TimelineRequest, req: Request):
    try:
        ms = generate_timeline(case_type=request.case_type, facts=request.facts, outcome=request.outcome)
        return TimelineResponse(milestones=ms)
    except Exception:
        raise _internal_error("/timeline", req)

@router.get("/health", tags=["Health"])
async def health():
    from backend.core.cache import cache_stats
    return {"status":"ok","service":"AI Legal Advisor India","version":"3.0.0","cache":cache_stats()}
