# ============================================================
# Dockerfile — RAG Legal Assistant (Corporate Law AI)
# ============================================================
# Multi-purpose build optimised for Render free tier:
#   • CPU-only PyTorch (~200 MB vs ~2 GB with CUDA)
#   • Embedding model baked into image (no cold-start download)
#   • Slim Python base to reduce image size
# ============================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Reduce glibc memory fragmentation (important for 512 MB RAM)
    MALLOC_ARENA_MAX=2 \
    # HuggingFace cache inside the image
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch first (saves ~1.5 GB) ──────────
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# ── Install Python dependencies ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model at build time ──────────
# This avoids a 30-60s download on every cold start.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ── Remove build dependencies to shrink image ───────────────
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ── Copy application code ───────────────────────────────────
COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/

COPY ui/ ./ui/
COPY data/vectorstore/ ./data/vectorstore/

# Create the documents directory if it is not present
RUN mkdir -p /app/data/documents



# ── Create data directory for analytics DB ───────────────────
RUN mkdir -p /app/data

# ── Expose port (Render sets $PORT dynamically) ─────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start the server ────────────────────────────────────────
# Use shell form so $PORT is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120
