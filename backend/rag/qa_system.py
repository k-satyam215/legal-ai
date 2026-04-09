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

    # 🔥 CONTEXT CLEANING (HALLUCINATION FIX)
    filtered_docs = []
    for doc in docs:
        text = doc.page_content

        # Remove noisy legal citations
        if "AIR" in text or "SCR" in text:
            continue

        # Trim long text
        if len(text) > 1000:
            text = text[:1000]

        if text.strip():
            filtered_docs.append(text)

    context = "\n\n".join(filtered_docs)

    # 🔥 FINAL PRODUCTION PROMPT
    prompt = f"""
You are a professional Indian lawyer with real courtroom experience.

Use ONLY the context below.

Context:
{context}

User Question:
{query}

IMPORTANT RULES:
- Return ONLY valid JSON (no markdown, no extra text)
- Give practical legal reasoning (not generic)
- Analyze BOTH sides (tenant and landlord if applicable)
- Prefer the MOST relevant SINGLE law
- Mention exact section if possible (e.g., Section 108, Transfer of Property Act, 1882)
- Explain practical implications (notice period, deduction vs adjustment of deposit)
- Use correct legal terminology (avoid "forfeit", prefer "deduct" or "adjust")
- DO NOT assume facts not given in the question
- DO NOT include random case details (rent, duration, etc.)
- If facts are unclear, give general legal principle-based answer
- Steps must follow this structure:
  1. Communication / Legal Notice
  2. Evidence Collection
  3. Legal Action

Case types:
- rent
- fraud
- job
- other

If case is NOT rent → notice_points = []

Format:

{{
  "case_type": "rent | fraud | job | other",
  "law": "most relevant law with section",
  "explanation": "clear practical explanation in Hinglish (include both sides)",
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "notice_points": [
    "You are hereby called upon to refund the security deposit amount of Rs. [amount] within 15 days from the receipt of this notice, failing which legal proceedings shall be initiated against you.",
    "As per the terms of the rent agreement and applicable laws, you are legally bound to return the deposit amount after deduction of legitimate charges, if any.",
    "Your failure to refund the said amount constitutes a breach of contractual obligations.",
    "In case of non-compliance, the tenant reserves the right to initiate legal proceedings including a recovery suit and claim for damages.",
    "You are advised to comply within the stipulated time to avoid legal consequences."
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

    # ✅ Clean markdown safely
    response_text = response_text.strip()
    response_text = response_text.replace("```json", "").replace("```", "")

    # ✅ Safe JSON extraction
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
            "explanation": "System could not process the response properly. Please try again.",
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