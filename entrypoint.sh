#!/bin/bash
set -e

mkdir -p backend/rag

# Download large index files from Google Drive if not already present.
# File IDs come from environment variables (set as HF Space "Variables" — not secret, but harmless if public).
download_if_missing() {
  local filepath="$1"
  local file_id="$2"
  if [ ! -f "$filepath" ]; then
    echo "[entrypoint] Downloading $filepath ..."
    gdown --id "$file_id" -O "$filepath"
  else
    echo "[entrypoint] $filepath already present, skipping download."
  fi
}

download_if_missing "backend/rag/faiss_index.index" "$FAISS_INDEX_FILE_ID"
download_if_missing "backend/rag/faiss_index_metadata.json" "$FAISS_METADATA_FILE_ID"
download_if_missing "backend/rag/faiss_index_offsets.npy" "$FAISS_OFFSETS_FILE_ID"
download_if_missing "backend/rag/bm25_index.pkl" "$BM25_INDEX_FILE_ID"

echo "[entrypoint] All data files ready. Starting Streamlit..."
exec streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
