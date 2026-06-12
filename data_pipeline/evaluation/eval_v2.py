"""eval_v2.py — Evaluation for Legal Advisor v3.
Usage: python data_pipeline/evaluation/eval_v2.py --verbose
"""
import sys, os, json, time, argparse, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.rag.loader import initialize
from backend.v2.router_v2 import route_query_v2
initialize()
logging.basicConfig(level=logging.WARNING)
TEST_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")

def run_evaluation(verbose=False):
    with open(TEST_PATH) as f: tcs = json.load(f)
    print(f"\n{'='*55}\n🧪 Eval v3 — {len(tcs)} test cases\n{'='*55}")
    results = []
    for tc in tcs:
        query = tc["query"]; t0 = time.time()
        try:
            r   = route_query_v2(query)
            ms  = (time.time()-t0)*1000
            law = r.get("analysis","") + " ".join(r.get("laws",[]))
            kws = tc["expected_law_keywords"]
            kwr = sum(1 for k in kws if k.lower() in law.lower())/len(kws) if kws else 1.0
            # Check section enforcement
            has_sec = any(import_re_findall(l) for l in r.get("laws",[]))
            row = {
                "id":tc["id"],"query":query[:55],
                "case_type_ok": r.get("case_type")==tc["expected_case_type"],
                "kw_rate": round(kwr,2),
                "steps_ok": len(r.get("steps",[]))>=tc.get("min_steps",1),
                "notice_ok": r.get("notice_applicable")==tc["expected_notice_applicable"],
                "risk_ok": r.get("risk_level")==tc["expected_risk_level"],
                "has_issue": bool(r.get("issue","")),
                "has_strategy": bool(r.get("strategy","")),
                "has_followup": len(r.get("follow_up_questions",[]))>=2,
                "latency_ms": round(ms,0), "error": None,
            }
            if verbose:
                s = "✅" if row["case_type_ok"] else "❌"
                print(f"\n{s} [{tc['id']}] {query[:55]}")
                print(f"   ct={r.get('case_type')} kw={kwr:.0%} laws={r.get('laws',[])} ms={ms:.0f}")
        except Exception as e:
            row = {"id":tc["id"],"query":query[:55],"case_type_ok":False,"kw_rate":0.0,
                   "steps_ok":False,"notice_ok":False,"risk_ok":False,"has_issue":False,
                   "has_strategy":False,"has_followup":False,"latency_ms":0,"error":str(e)}
            if verbose: print(f"💥 [{tc['id']}] {e}")
        results.append(row)
    n   = len(results)
    lts = sorted(r["latency_ms"] for r in results)
    s   = {
        "total": n,
        "case_type_accuracy":   round(sum(r["case_type_ok"] for r in results)/n,3),
        "kw_hit_rate":          round(sum(r["kw_rate"] for r in results)/n,3),
        "step_coverage":        round(sum(r["steps_ok"] for r in results)/n,3),
        "notice_accuracy":      round(sum(r["notice_ok"] for r in results)/n,3),
        "risk_accuracy":        round(sum(r["risk_ok"] for r in results)/n,3),
        "issue_present":        round(sum(r["has_issue"] for r in results)/n,3),
        "strategy_present":     round(sum(r["has_strategy"] for r in results)/n,3),
        "followup_present":     round(sum(r["has_followup"] for r in results)/n,3),
        "avg_latency_ms":       round(sum(lts)/n,1),
        "p50_ms":               lts[n//2], "p95_ms": lts[int(n*0.95)],
        "errors":               sum(1 for r in results if r["error"]),
    }
    print(f"\n{'='*55}\n📊 RESULTS v3\n{'='*55}")
    for k,v in s.items():
        e = "✅" if isinstance(v,float) and v>=0.8 else ("⚠️" if isinstance(v,float) and v>=0.6 else "📊")
        print(f"  {e}  {k:30s}: {v}")
    out = os.path.join(os.path.dirname(__file__), "eval_results_v3.json")
    with open(out,"w") as f: json.dump({"summary":s,"details":results},f,indent=2)
    print(f"\n💾 → {out}")

def import_re_findall(text):
    import re
    return bool(re.search(r"\d+", text))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--verbose",action="store_true")
    run_evaluation(verbose=p.parse_args().verbose)
