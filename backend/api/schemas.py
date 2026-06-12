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
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list)
    case_type: Optional[str] = Field(default="general")
    session_id: Optional[str] = Field(default="default")

class ChatResponse(BaseModel):
    reply: str
    quick_card: Optional[str] = None
    needs_deep_advice: bool = False
    suggested_action: Optional[str] = None
    detected_intent: Optional[str] = None

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
