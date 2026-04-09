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


# ✅ STEP 1: Ensure vector DB exists
def ensure_vector_db():
    if not os.path.exists("backend/rag/faiss_index"):
        print("⚡ Creating FAISS index (first run)...")
        from backend.rag.rag_pipeline import create_vector_db
        create_vector_db()


# ✅ STEP 2: Load retriever
def get_retriever():
    ensure_vector_db()

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

    # 🔥 STRONG PROFESSIONAL PROMPT
    prompt = f"""
You are a professional Indian lawyer with courtroom experience.

Use ONLY the context below.

Context:
{context}

User Question:
{query}

IMPORTANT RULES:
- Return ONLY JSON
- Give practical legal reasoning (not generic)
- Analyze BOTH sides (tenant and landlord)
- Mention when landlord can legally deduct deposit
- Highlight real-world dispute scenarios
- Steps must be actionable

Case types:
- rent
- fraud
- job
- other

If case is NOT rent → notice_points = []

Format:

{{
  "case_type": "rent | fraud | job | other",
  "law": "specific law with section",
  "explanation": "deep practical explanation in Hinglish",
  "steps": [
    "clear actionable step 1",
    "clear actionable step 2",
    "clear actionable step 3"
  ],
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
        temperature=0.2,
    )

    response_text = response.choices[0].message.content
    print("\n📦 RAW RESPONSE:\n", response_text)

    # ✅ Clean markdown safely (IMPORTANT FIX)
    response_text = response_text.strip()
    response_text = response_text.replace("```json", "").replace("```", "")

    # ✅ Safe JSON extraction (FIXED)
    try:
        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            json_text = match.group()
            data = json.loads(json_text)
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print("❌ JSON parse error:", e)
        print("🔍 RAW RESPONSE:", response_text)

        return {
            "case_type": "other",
            "law": "Parsing Error",
            "explanation": "System could not properly process the response. Please try again.",
            "steps": [],
            "notice_points": []
        }

    return data


# ✅ CLI test
if __name__ == "__main__":
    q = input("Enter your legal question: ")
    ans = ask_question(q)

    print("\n💡 FINAL OUTPUT:\n")
    print(ans)