from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any

class AskRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    language: Optional[str] = Field(default="en")

class LegalAdviceResponse(BaseModel):
    issue: str
    case_type: str
    laws: list[str]
    analysis: str
    steps: list[str]
    risk_level: str
    strategy: str
    notice_applicable: bool
    follow_up_questions: list[str]
    classification: dict
    request_id: Optional[str] = None

class DeepAskRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    extra_context: Optional[str] = Field(default="", max_length=500)
    language: Optional[str] = Field(default="en")

class DeepAnalysisResponse(BaseModel):
    issue: str
    case_type: str
    primary_law: str
    laws: list[str]
    legal_interpretation: str
    scenario_analysis: Any
    analysis: str
    steps: list[str]
    alternative_remedies: list[str]
    risk_level: str
    risk_factors: list[str]
    strategy: str
    timeline_estimate: str
    notice_applicable: bool
    follow_up_questions: list[str]
    classification: dict
    request_id: Optional[str] = None

class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)

class ClassifyResponse(BaseModel):
    case_type: str
    confidence: float
    reason: str

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    case_type: Optional[str] = Field(default="general")
    # No shared default here on purpose: routes.py generates a fresh random
    # session_id per request when the client omits one. A shared literal
    # default (e.g. "default") would put every client that doesn't track
    # sessions into the SAME conversation memory bucket — a real cross-user
    # data leak on this endpoint (unrelated users' facts/history mixing).
    session_id: Optional[str] = Field(default=None, max_length=100)

    @field_validator("message")
    @classmethod
    def message_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value

class ChatResponse(BaseModel):
    reply: str
    quick_card: Optional[str] = None
    needs_deep_advice: bool = False
    suggested_action: Optional[str] = None
    detected_intent: Optional[str] = None
    # Echoed back so a client that didn't send one can persist it and reuse
    # it on the next turn to keep continuity in the SAME session (see the
    # comment on ChatRequest.session_id for why there's no shared default).
    session_id: Optional[str] = None

class NoticeRequest(BaseModel):
    notice_type: str
    sender_name: str
    sender_address: str
    recipient_name: str
    recipient_address: str
    facts: str = Field(..., min_length=20, max_length=1000)
    relief: str
    law: str
    generate_pdf: bool = False

    @field_validator("notice_type")
    @classmethod
    def validate_notice_type(cls, v):
        valid = {"deposit_refund","eviction","consumer_complaint","employment_termination","general"}
        if v not in valid: raise ValueError(f"Must be one of {valid}")
        return v

class NoticeResponse(BaseModel):
    notice_text: str
    pdf_path: Optional[str] = None

class TimelineRequest(BaseModel):
    case_type: str
    facts: str
    outcome: str

class TimelineResponse(BaseModel):
    milestones: list[dict]
