"""
FastAPI backend for the RAG Legal Assistant.

Endpoints
─────────
  POST /ask      — answer a legal question with source citations
  POST /ingest   — (re)build the FAISS index from PDFs in data/documents/

Startup behaviour
─────────────────
  On application boot we attempt to pre-load the FAISS index into memory
  so the first /ask request doesn't pay the deserialization cost.  If the
  index doesn't exist yet, the app still starts — /ingest must be called
  first.

Run locally:
  uvicorn app.main:app --reload --port 8000
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# ── Make src/ and app/ importable ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from app.schemas import AskRequest, AskResponse, IngestResponse    # noqa: E402
from rag_chain import ask                                          # noqa: E402
from vectorstore import load_vectorstore, create_vectorstore       # noqa: E402

logger = logging.getLogger("rag_legal_assistant")


# ── Startup / shutdown lifecycle ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the FAISS index at startup (if one exists)."""
    try:
        load_vectorstore()
        logger.info("FAISS index loaded at startup.")
    except FileNotFoundError:
        logger.warning(
            "No FAISS index found — call POST /ingest before querying."
        )
    yield                           # app runs
    # (nothing to clean up)


# ── Configuration ────────────────────────────────────────────────

# Origins allowed to call the API.  Read from ALLOWED_ORIGINS env var
# (comma-separated) so the same code works locally and on Render.
# The UI is served from the same origin via /ui, so it is always allowed.
_default_origins = "http://localhost,http://localhost:8000,http://127.0.0.1,http://127.0.0.1:8000"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]


# ── App instance ─────────────────────────────────────────────────

app = FastAPI(
    title="RAG Legal Assistant",
    version="1.0.0",
    description="Ask questions over legal PDFs and get answers with source citations.",
    lifespan=lifespan,
)

# ── CORS middleware (must be added before routes are resolved) ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# ── Serve the frontend UI from ui/ ──────────────────────────────
UI_DIR = PROJECT_ROOT / "ui"
if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


# ── Endpoints ────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask_question(body: AskRequest):
    """Answer a legal question using retrieved PDF context."""
    try:
        result = ask(body.question)
        return AskResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "FAISS index not found. "
                "Call POST /ingest first to build the index."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Error during /ask")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {exc}",
        ) from exc


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """Rebuild the FAISS index from all PDFs in data/documents/."""
    try:
        vs = create_vectorstore()
        return IngestResponse(
            status="success",
            chunks_indexed=vs.index.ntotal,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No PDFs found: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Error during /ingest")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc


@app.get("/health")
async def health():
    """Simple health check for Render / monitoring."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Redirect the bare root to the chat UI."""
    return RedirectResponse(url="/ui/index.html")
