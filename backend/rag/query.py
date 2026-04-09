from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def query_rag(question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local("backend/rag/faiss_index", embeddings)

    docs = db.similarity_search(question, k=3)

    return docs


if __name__ == "__main__":
    results = query_rag("landlord deposit issue")

    for r in results:
        print(r.page_content)