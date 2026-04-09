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


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "backend/rag/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": 3})


def ask_question(query):
    retriever = get_retriever()
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # 🔥 SMART PROMPT (multi-case)
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
  "explanation": "simple explanation",
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

    # Clean markdown
    response_text = response_text.strip()
    response_text = response_text.replace("```json", "").replace("```", "")

    try:
        json_text = re.search(r"\{.*\}", response_text, re.DOTALL).group()
        data = json.loads(json_text)
    except Exception as e:
        print("❌ JSON parse error:", e)
        return response_text

    return data