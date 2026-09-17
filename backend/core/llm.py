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

# reasoning_effort is only accepted by Groq's GPT-OSS models (20b/120b).
# Guard it so switching LLM_MODEL to a non-reasoning model doesn't crash every call.
_REASONING_CAPABLE = ("gpt-oss",)

def _supports_reasoning_effort(model: str) -> bool:
    return any(tag in model for tag in _REASONING_CAPABLE)

def call_llm(messages, model=LLM_MODEL, temperature=LLM_TEMPERATURE,
             max_tokens=LLM_MAX_TOKENS, retries=3, backoff=2.0) -> str:
    client = get_client()
    use_reasoning = _supports_reasoning_effort(model)
    for attempt in range(1, retries+1):
        try:
            kwargs = dict(model=model, messages=messages,
                          temperature=temperature, max_tokens=max_tokens)
            if use_reasoning:
                kwargs["reasoning_effort"] = "low"
            r = client.chat.completions.create(**kwargs)
            content = r.choices[0].message.content
            if not content:
                raise RuntimeError(f"Empty response from model (finish_reason={r.choices[0].finish_reason})")
            return content
        except TypeError as e:
            # SDK/model rejected a kwarg (e.g. old SDK, or model without reasoning support).
            # Drop reasoning_effort once and retry immediately instead of burning all retries.
            if use_reasoning and "reasoning_effort" in str(e):
                logger.warning(f"[LLM] reasoning_effort not accepted, retrying without it: {e}")
                use_reasoning = False
                continue
            logger.warning(f"[LLM] Attempt {attempt}/{retries}: {e}")
            if attempt < retries: time.sleep(backoff**attempt)
            else: raise RuntimeError(f"LLM failed after {retries} retries: {e}") from e
        except Exception as e:
            logger.warning(f"[LLM] Attempt {attempt}/{retries}: {e}")
            if attempt < retries: time.sleep(backoff**attempt)
            else: raise RuntimeError(f"LLM failed after {retries} retries: {e}") from e
    return ""
