"""
memory.py v4 — Structured case state memory.
Stores: issue, case_type, facts, laws_discussed, last_actions.
Smart reset on topic change.
"""
import time, hashlib, threading, re
from dataclasses import dataclass, field
from typing import Optional

SESSION_TTL  = 1800
MAX_TURNS    = 8
MAX_SESSIONS = 300

_CASE_KW: dict[str, list[str]] = {
    "rent":       ["rent","deposit","landlord","tenant","evict","flat","kiraya","makan"],
    "consumer":   ["product","refund","amazon","flipkart","consumer","defective","delivery","warranty"],
    "criminal":   ["fir","police","fraud","cheat","stolen","chori","thagi","bail","arrest","cyber","otp"],
    "employment": ["salary","job","fired","employer","wages","naukri","pf","retrenchment","layoff"],
    "general":    ["noise","neighbour","society","water","parking","documents","rwa"],
}

def _detect_topic(text: str) -> str:
    t = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in _CASE_KW.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"

def _topics_differ(t1: str, t2: str) -> bool:
    if t1 == t2: return False
    if "general" in (t1, t2): return False
    return True


@dataclass
class CaseState:
    """Structured case memory — not just raw messages."""
    issue: str = ""
    case_type: str = "general"
    facts: list[str] = field(default_factory=list)
    laws_discussed: list[str] = field(default_factory=list)
    last_actions: list[str] = field(default_factory=list)
    amounts_mentioned: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Format structured state for LLM prompt injection."""
        parts = []
        if self.issue:
            parts.append(f"Established issue: {self.issue}")
        if self.case_type != "general":
            parts.append(f"Case type: {self.case_type}")
        if self.facts:
            parts.append(f"Known facts: {'; '.join(self.facts[-3:])}")
        if self.laws_discussed:
            parts.append(f"Laws already discussed: {', '.join(self.laws_discussed[-3:])}")
        if self.last_actions:
            parts.append(f"Previous advice: {'; '.join(self.last_actions[-2:])}")
        return " | ".join(parts) if parts else ""


@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    topic: str = "general"


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    current_topic: str = "general"
    case_state: CaseState = field(default_factory=CaseState)
    turn_count: int = 0


class MemoryStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _evict(self):
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if now - s.last_active > SESSION_TTL]
            for sid in expired: del self._sessions[sid]
            if len(self._sessions) > MAX_SESSIONS:
                oldest = sorted(self._sessions.items(), key=lambda x: x[1].last_active)
                for sid, _ in oldest[:len(self._sessions) - MAX_SESSIONS]: del self._sessions[sid]

    def get_or_create(self, session_id: str) -> Session:
        self._evict()
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
            s = self._sessions[session_id]
            s.last_active = time.time()
            return s

    def add_turn(self, session_id: str, role: str, content: str, laws: Optional[list[str]] = None) -> None:
        s = self.get_or_create(session_id)
        topic = _detect_topic(content)
        with self._lock:
            # Smart reset on topic change
            if role == "user" and s.turns and _topics_differ(topic, s.current_topic):
                s.turns = s.turns[-2:]
                s.case_state = CaseState()
                s.current_topic = topic

            s.turns.append(Turn(role=role, content=content, topic=topic))
            s.turn_count += 1

            if topic != "general": s.current_topic = topic
            if laws:
                for l in laws:
                    if l not in s.case_state.laws_discussed:
                        s.case_state.laws_discussed.append(l)

            if len(s.turns) > MAX_TURNS:
                s.turns = s.turns[-MAX_TURNS:]

    def update_case_state(self, session_id: str, issue: str = "", facts: list[str] = None,
                          actions: list[str] = None, amounts: list[str] = None) -> None:
        """Update structured case state from LLM response."""
        s = self.get_or_create(session_id)
        with self._lock:
            if issue: s.case_state.issue = issue
            if facts:
                for f in facts:
                    if f not in s.case_state.facts: s.case_state.facts.append(f)
            if actions:
                for a in actions:
                    if a not in s.case_state.last_actions: s.case_state.last_actions.append(a)
            if amounts:
                for amt in amounts:
                    if amt not in s.case_state.amounts_mentioned: s.case_state.amounts_mentioned.append(amt)
            # Keep lists bounded
            s.case_state.facts = s.case_state.facts[-5:]
            s.case_state.last_actions = s.case_state.last_actions[-3:]

    def get_context(self, session_id: str, max_turns: int = 4) -> dict:
        s = self.get_or_create(session_id)
        recent = s.turns[-max_turns:]
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            content = t.content[:260] + "..." if len(t.content) > 260 else t.content
            lines.append(f"{prefix}: {content}")
        return {
            "history_text":   "\n".join(lines) if lines else "None",
            "case_type":      s.current_topic,
            "case_state":     s.case_state.to_context_string(),
            "laws_discussed": s.case_state.laws_discussed[:4],
            "is_followup":    len(s.turns) > 2,
            "turn_count":     s.turn_count,
        }

    def clear(self, session_id: str) -> None:
        with self._lock: self._sessions.pop(session_id, None)

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            return {"total": len(self._sessions),
                    "active": sum(1 for s in self._sessions.values() if now - s.last_active < 300)}


_store = MemoryStore()

def get_memory() -> MemoryStore: return _store

def make_session_id(ip: str = "", user_agent: str = "") -> str:
    return hashlib.md5(f"{ip}|{user_agent}|{int(time.time() // SESSION_TTL)}".encode()).hexdigest()[:12]

def format_history_for_llm(history: list[dict], max_turns: int = 4) -> str:
    if not history: return "None"
    recent = history[-max_turns:]
    lines = []
    for t in recent:
        role = "User" if t.get("role") == "user" else "Assistant"
        content = t.get("content", "")
        if len(content) > 220: content = content[:220] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def is_followup(message: str, history: list[dict]) -> bool:
    if not history: return False
    signals = ["aur","or","also","then","phir","uske baad","what about","kya","how",
               "kaise","kitna","documents","papers","kahan","next","baad","iske alawa"]
    m = message.lower()
    return len(message.split()) < 9 and any(s in m for s in signals)
