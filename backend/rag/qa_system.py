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
You are a professional Indian lawyer with practical courtroom experience.

Use ONLY the context below.

Context:
{context}

User Question:
{query}

IMPORTANT RULES:
- Return ONLY valid JSON
- No markdown or extra text
- Give PRACTICAL legal advice (not generic)
- Mention specific laws (like Contract Act, Transfer of Property Act)
- Explain BOTH sides (tenant & landlord if applicable)
- Steps must be actionable and realistic

Case types:
- rent
- fraud
- job
- other

If case is NOT rent → return empty notice_points []

Format:

{{
  "case_type": "rent | fraud | job | other",
  "law": "specific Indian law with section if possible",
  "explanation": "clear practical explanation in Hinglish",
  "steps": [
    "clear actionable step 1",
    "clear actionable step 2",
    "clear actionable step 3"
  ],
  "notice_points": [
    "You are hereby called upon to refund the security deposit amount of Rs. [amount] within 15 days from the receipt of this notice, failing which legal proceedings shall be initiated against you.",
    "As per the terms of the rent agreement and applicable laws, you are legally bound to return the deposit amount after deduction of legitimate charges, if any.",
    "Your failure to refund the said amount constitutes a breach of contract and unlawful retention of funds.",
    "In case of non-compliance, the tenant reserves the right to initiate legal proceedings including recovery suit and claim for damages.",
    "You are advised to comply within the stipulated time to avoid legal consequences."
  ]
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # 🔥 more stable output
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
            "explanation": "System could not generate proper legal advice.",
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