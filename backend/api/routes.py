"""API Routes — /api/v2"""
import re, logging, tempfile, os
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from backend.v2.router_v2 import route_query_v2, route_deep_analysis
from backend.v2.classifier_v2 import classify_query_v2
from backend.v2.chat_engine import chat_response
from backend.services.notice_generator import generate_notice_text, generate_notice_pdf
from backend.services.timeline_generator import generate_timeline
from backend.api.schemas import (
    AskRequest, LegalAdviceResponse, DeepAskRequest, DeepAnalysisResponse,
    ClassifyRequest, ClassifyResponse, ChatRequest, ChatResponse,
    NoticeRequest, NoticeResponse, TimelineRequest, TimelineResponse,
)
logger = logging.getLogger(__name__)
router = APIRouter()

def _s(text: str) -> str:
    return " ".join(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]","",text).split()).strip()

@router.post("/ask", response_model=LegalAdviceResponse, tags=["Legal Advisor"])
async def ask(request: AskRequest, req: Request):
    try:
        q = _s(request.query)
        if not q: raise HTTPException(status_code=422, detail="Empty query")
        return route_query_v2(q, language=request.language or "en")
    except HTTPException: raise
    except Exception as e:
        rid = getattr(getattr(req,"state",None),"request_id","?")
        logger.error(f"[/ask] {e}")
        raise HTTPException(status_code=500, detail={"error":str(e),"request_id":rid})

@router.post("/deep-ask", response_model=DeepAnalysisResponse, tags=["Legal Advisor"])
async def deep_ask(request: DeepAskRequest, req: Request):
    try:
        q = _s(request.query)
        if not q: raise HTTPException(status_code=422, detail="Empty query")
        return route_deep_analysis(q, extra_context=request.extra_context or "", language=request.language or "en")
    except HTTPException: raise
    except Exception as e:
        rid = getattr(getattr(req,"state",None),"request_id","?")
        logger.error(f"[/deep-ask] {e}")
        raise HTTPException(status_code=500, detail={"error":str(e),"request_id":rid})

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    try:
        msg     = _s(request.message)
        history = [{"role":m.role,"content":m.content} for m in request.history]
        result  = chat_response(msg, history, case_type=request.case_type or "general", session_id=request.session_id or "default")
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"[/chat] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/classify-case", response_model=ClassifyResponse, tags=["Classification"])
async def classify_case(request: ClassifyRequest):
    try: return classify_query_v2(_s(request.query))
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-notice", response_model=NoticeResponse, tags=["Notice"])
async def generate_notice_api(request: NoticeRequest, background_tasks: BackgroundTasks):
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
    except Exception as e:
        logger.error(f"[/notice] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/timeline", response_model=TimelineResponse, tags=["Timeline"])
async def get_timeline(request: TimelineRequest):
    try:
        ms = generate_timeline(case_type=request.case_type, facts=request.facts, outcome=request.outcome)
        return TimelineResponse(milestones=ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", tags=["Health"])
async def health():
    from backend.core.cache import cache_stats
    return {"status":"ok","service":"AI Legal Advisor India","version":"3.0.0","cache":cache_stats()}
