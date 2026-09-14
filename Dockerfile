# ── AI Legal Advisor India — HuggingFace Spaces (single-service) ─────────
FROM python:3.11-slim

WORKDIR /app

# System deps for faiss / sentence-transformers / pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

# Run as a non-root user — same reasoning as Dockerfile.backend. Done after
# chmod so entrypoint.sh keeps its executable bit, and chown covers
# backend/rag/ so entrypoint.sh (running as appuser) can still write the
# FAISS index files it downloads at container startup.
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["./entrypoint.sh"]
