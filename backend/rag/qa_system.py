import os
import json
import re
import sys
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from notice_generator import generate_notice

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ✅ STEP 1: Ensure vector DB exists (ONLY when needed)
def ensure_vector_db():
    if not os.path.exists("backend/rag/faiss_index"):
        print("⚡ Creating FAISS index (first run)...")
        from backend.rag.rag_pipeline import create_vector_db
        create_vector_db()


# ✅ STEP 2: Load retriever
def get_retriever():
    ensure_vector_db()  # 🔥 important

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "backend/rag/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": 3})


# ✅ STEP 3: Ask question
def ask_question(query):
    retriever = get_retriever()
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert Indian legal advisor.

Use ONLY the context below.

Context:
{context}

User Question:
{query}

IMPORTANT:
- Return ONLY JSON
- No markdown or extra text
- Detect case type

Case types:
- rent
- fraud
- job
- other

If case is NOT rent → return empty notice_points []

Format:

{{
  "case_type": "rent | fraud | job | other",
  "law": "relevant Indian law",
  "explanation": "simple explanation (Hindi + English mix)",
  "steps": ["step1", "step2"],
  "notice_points": [
    "legal sentence 1",
    "legal sentence 2",
    "legal sentence 3",
    "legal sentence 4",
    "legal sentence 5"
  ]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    response_text = response.choices[0].message.content
    print("\n📦 RAW RESPONSE:\n", response_text)

    # ✅ Clean markdown safely
    response_text = response_text.strip()
    response_text = re.sub(r"```.*?```", "", response_text, flags=re.DOTALL)

    # ✅ Safe JSON extraction
    try:
        json_text = re.search(r"\{.*\}", response_text, re.DOTALL).group()
        data = json.loads(json_text)
    except Exception as e:
        print("❌ JSON parse error:", e)
        return {
            "case_type": "other",
            "law": "N/A",
            "explanation": "Could not process response properly.",
            "steps": [],
            "notice_points": []
        }

    return data


# ✅ CLI testing (optional)
if __name__ == "__main__":
    q = input("Enter your legal question: ")
    ans = ask_question(q)

    print("\n💡 FINAL OUTPUT:\n")
    print(ans)