"""router_v2.py — Main orchestrator with cache."""
import time, uuid, logging
from backend.v2.classifier_v2 import classify_query_v2, _score_rules, _resolve
from backend.v2.legal_advisor_v2 import get_legal_advice_v2, get_deep_analysis
from backend.core.cache import cache_get, cache_set
logger = logging.getLogger(__name__)

def route_query_v2(query: str, language: str = "en") -> dict:
    rid = str(uuid.uuid4())[:8]; t0 = time.perf_counter()
    logger.info(f"[Router:{rid}] '{query[:80]}'")
    scores = _score_rules(query); quick, _ = _resolve(scores)
    cached = cache_get(query, quick or "unknown")
    if cached:
        cached["request_id"] = rid
        logger.info(f"[Router:{rid}] CACHE HIT {(time.perf_counter()-t0)*1000:.0f}ms")
        return cached
    clf = classify_query_v2(query); ct = clf["case_type"]
    advice = get_legal_advice_v2(query=query, case_type=ct)
    advice["classification"] = clf; advice["request_id"] = rid
    cache_set(query, advice, case_type=ct, ttl=3600)
    logger.info(f"[Router:{rid}] TOTAL {(time.perf_counter()-t0)*1000:.0f}ms")
    return advice

def route_deep_analysis(query: str, extra_context: str = "", language: str = "en") -> dict:
    rid = str(uuid.uuid4())[:8]; t0 = time.perf_counter()
    clf = classify_query_v2(query); ct = clf["case_type"]
    result = get_deep_analysis(query=query, case_type=ct, extra_context=extra_context)
    result["classification"] = clf; result["request_id"] = rid
    logger.info(f"[DeepRouter:{rid}] TOTAL {(time.perf_counter()-t0)*1000:.0f}ms")
    return result
