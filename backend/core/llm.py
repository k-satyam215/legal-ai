import os, time, logging
from groq import Groq
from backend.core.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
logger = logging.getLogger(__name__)
_client: Groq | None = None

def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def call_llm(messages, model=LLM_MODEL, temperature=LLM_TEMPERATURE,
             max_tokens=LLM_MAX_TOKENS, retries=3, backoff=2.0) -> str:
    client = get_client()
    for attempt in range(1, retries+1):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            logger.warning(f"[LLM] Attempt {attempt}/{retries}: {e}")
            if attempt < retries: time.sleep(backoff**attempt)
            else: raise RuntimeError(f"LLM failed after {retries} retries: {e}") from e
    return ""
