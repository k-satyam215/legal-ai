"""CLI: test retrieval from built index."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.rag.loader import initialize
from backend.v2.query_understanding import understand_query
from backend.v2.smart_retriever import smart_retrieve


def query_index(query: str, k: int = 5):
    initialize()
    print(f"\n🔍 Query: {query}\n{'─'*60}")
    ctx  = understand_query(query)
    docs = smart_retrieve(ctx, final_k=k)
    print(f"📊 Retrieved: {len(docs)} documents\n")
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        print(f"[{i}] {meta.get('law_name','N/A')} | {meta.get('section','N/A')} | score={doc.get('final_score',0):.3f}")
        print(f"    {doc.get('text','')[:200]}...")
        print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "landlord not returning security deposit"
    query_index(q)
