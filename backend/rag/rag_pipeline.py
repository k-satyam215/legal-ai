import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Absolute path (Windows safe)
DATA_PATH = os.path.abspath("backend/data")


def load_documents():
    docs = []

    print(f"📂 Reading data from: {DATA_PATH}")

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            # 🔥 FIX: handle .PDF / .pdf
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)

                try:
                    pdf = fitz.open(path)

                    for page in pdf:
                        # 🔥 better extraction
                        text = page.get_text("text")

                        # skip empty text
                        if text and text.strip():
                            docs.append(text)

                except Exception as e:
                    print(f"❌ Error reading {path}: {e}")

    return docs


def create_vector_db():
    raw_docs = load_documents()

    print(f"\n📄 Total raw docs: {len(raw_docs)}")

    # ❌ Stop if no data
    if len(raw_docs) == 0:
        print("❌ No text extracted. Check PDFs or folder path.")
        return

    # show sample
    print("\n🧾 Sample text preview:\n")
    print(raw_docs[0][:500])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.create_documents(raw_docs)

    print(f"\n🔹 Total chunks: {len(chunks)}")

    if len(chunks) == 0:
        print("❌ No chunks created.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\n⚡ Creating FAISS index...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("backend/rag/faiss_index")

    print("\n✅ Vector DB created successfully!")
    print("📁 Saved at: backend/rag/faiss_index")


if __name__ == "__main__":
    create_vector_db()