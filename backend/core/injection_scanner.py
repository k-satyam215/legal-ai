"""
backend/core/injection_scanner.py — ML-based prompt-injection detection via
LLM Guard (Protect AI's DeBERTa-based classifier).

This is a hard pre-LLM gate, layered on top of (not replacing) the two
defenses already in place: the regex heuristic in backend/core/security.py,
and the <user_query>/<retrieved_context> tag-wrapping + anti-injection
instructions in backend/core/prompts.py. Those two keep working even if
this module can't load the ML model at all.

Design goals, in order:
1. Never crash the app. Import failure, model-download failure, OOM, or a
   disabled ENABLE_ML_INJECTION_SCAN flag must all degrade to "scan
   unavailable" — the caller then falls through to the existing prompt-level
   defenses instead of blocking legitimate legal queries.
2. Load the model once per process (lazy singleton), not once per request —
   it's a ~700MB DeBERTa classifier; loading it per-call would be far too
   slow and memory-hungry.
3. CPU-only. HF Spaces free/basic tiers have no GPU, and forcing CPU avoids
   a slow/failed CUDA init attempt on hosts that report a GPU but can't
   actually use it from this process.
"""
import logging
import threading

from backend.core.config import ENABLE_ML_INJECTION_SCAN

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_scanner = None
_unavailable = not ENABLE_ML_INJECTION_SCAN


def _get_scanner():
    global _scanner, _unavailable
    if _scanner is not None or _unavailable:
        return _scanner
    with _lock:
        if _scanner is not None or _unavailable:
            return _scanner
        try:
            import os
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU
            from llm_guard.input_scanners import PromptInjection
            from llm_guard.input_scanners.prompt_injection import MatchType
            _scanner = PromptInjection(threshold=0.75, match_type=MatchType.FULL)
            logger.info("[InjectionScanner] LLM Guard PromptInjection model loaded")
        except Exception as e:
            logger.warning(
                f"[InjectionScanner] Unavailable ({type(e).__name__}: {e}) — "
                "falling back to regex-only + prompt-tag defenses."
            )
            _unavailable = True
    return _scanner


def scan(text: str) -> tuple[bool, float]:
    """
    Returns (is_injection, risk_score). Fails open — (False, 0.0) — on any
    error, including the scanner never having loaded, so a resource-limited
    deployment never has requests break because of this optional layer.
    """
    if not text:
        return False, 0.0
    scanner = _get_scanner()
    if scanner is None:
        return False, 0.0
    try:
        _, is_valid, risk_score = scanner.scan(text)
        return (not is_valid), float(risk_score)
    except Exception as e:
        logger.warning(f"[InjectionScanner] scan() failed, treating as clean: {e}")
        return False, 0.0
