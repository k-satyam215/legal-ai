import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
BASE_DIR           = Path(__file__).resolve().parents[2]
DATA_DIR           = BASE_DIR / "backend" / "data"
FAISS_INDEX_PATH   = str(BASE_DIR / "backend" / "rag" / "faiss_index")
BM25_INDEX_PATH    = str(BASE_DIR / "backend" / "rag" / "bm25_index.pkl")
GROQ_API_KEY: str  = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str     = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int    = 2048
EMBEDDING_MODEL: str   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_VECTOR: int  = 6
TOP_K_BM25: int    = 6
TOP_K_RERANK: int  = 3
CHUNK_SIZE: int    = 600
CHUNK_OVERLAP: int = 80
CASE_TYPES   = ["rent", "consumer", "criminal", "employment", "general"]
RISK_LEVELS  = ["low", "medium", "high"]
NOTICE_TYPES = ["deposit_refund","eviction","consumer_complaint","employment_termination","general"]
