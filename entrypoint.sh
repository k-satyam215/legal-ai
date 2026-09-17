#!/bin/bash
set -e

mkdir -p backend/rag

# Large RAG index files are hosted on a Hugging Face Hub *dataset* repo instead of
# Google Drive. Google Drive enforces an undocumented per-file daily download quota
# ("Cannot retrieve the public link... or have had many accesses") that gets tripped
# almost every time this container restarts/rebuilds — HF Hub doesn't have that problem.
#
# Set HF_DATA_REPO as an HF Space "Variable" (not secret) to override the default,
# e.g. HF_DATA_REPO=yourname/legal-ai-rag-data
# If the dataset repo is private, HF_TOKEN (Space secret) is picked up automatically
# by huggingface-cli.
HF_DATA_REPO="${HF_DATA_REPO:-Satyam970/legal-ai-rag-data}"

download_if_missing() {
  local filepath="$1"
  local filename="$2"
  if [ ! -f "$filepath" ]; then
    echo "[entrypoint] Downloading $filename from HF dataset '$HF_DATA_REPO' ..."
    hf download "$HF_DATA_REPO" "$filename" \
      --repo-type dataset \
      --local-dir backend/rag
  else
    echo "[entrypoint] $filepath already present, skipping download."
  fi
}

download_if_missing "backend/rag/faiss_index.index" "faiss_index.index"
download_if_missing "backend/rag/faiss_index_metadata.json" "faiss_index_metadata.json"
download_if_missing "backend/rag/faiss_index_offsets.npy" "faiss_index_offsets.npy"
download_if_missing "backend/rag/bm25_index.pkl" "bm25_index.pkl"

echo "[entrypoint] All data files ready. Starting Streamlit..."
exec streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
