"""
FastAPI backend for the Corporate Law Assistant.

Endpoints
─────────
  GET  /                   — redirect to the chat UI
  GET  /health             — health check (public)
  GET  /api/auth/config    — Supabase public config (public)
  GET  /docs/{filename}    — serve a legal PDF (public, supports #page=N)
  POST /ask                — answer a legal question (auth required)
  POST /ingest             — rebuild FAISS index (admin, not exposed in UI)
  POST /auto-ingest        — incremental ingest (admin, not exposed in UI)
  GET  /api/analytics      — usage analytics (auth required)

Startup behaviour
─────────────────
  On boot: load the FAISS index + embedding model into memory,
  initialise the analytics database, and optionally start the
  auto-ingest background scheduler.

Run locally:
  uvicorn app.main:app --reload --port 8000
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Make src/, app/, and scripts/ importable ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "app", "scripts"):
    _p = str(PROJECT_ROOT / sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Configure logging before other imports ───────────────────────
from src.config import LOG_LEVEL  # noqa: E402

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag_legal_assistant")

# ── Application imports ──────────────────────────────────────────
from app.schemas import (  # noqa: E402
    AskRequest,
    AskResponse,
    IngestResponse,
    AutoIngestResponse,
    AuthConfigResponse,
    AnalyticsResponse,
)
from app.auth import get_current_user, AUTH_ENABLED  # noqa: E402
from app.analytics import (  # noqa: E402
    track_question,
    track_activity,
    get_analytics,
)
from src.config import (  # noqa: E402
    AUTO_INGEST_INTERVAL_HOURS,
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    ALLOWED_ORIGINS,
    DOCUMENTS_DIR,
)

_scheduler = None  # module-level reference so we can shut it down


# ── Startup / shutdown lifecycle ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the FAISS index and start the auto-ingest scheduler."""
    global _scheduler

    # 1. Pre-load FAISS index
    try:
        from src.vectorstore import load_vectorstore
        load_vectorstore()
        logger.info("FAISS index loaded at startup.")
    except FileNotFoundError:
        logger.warning(
            "No FAISS index found — call POST /ingest or /auto-ingest first."
        )
    except Exception as exc:
        logger.error("Failed to load FAISS index: %s", exc)

    # 2. Start background scheduler (if interval > 0)
    if AUTO_INGEST_INTERVAL_HOURS > 0:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from auto_ingest import run_pipeline

            _scheduler = BackgroundScheduler(daemon=True)
            _scheduler.add_job(
                lambda: run_pipeline(skip_scrape=False),
                trigger="interval",
                hours=AUTO_INGEST_INTERVAL_HOURS,
                id="auto_ingest",
                name="Periodic auto-ingest",
                replace_existing=True,
            )
            _scheduler.start()
            logger.info(
                "Auto-ingest scheduler started — every %dh",
                AUTO_INGEST_INTERVAL_HOURS,
            )
        except Exception as exc:
            logger.warning("Could not start auto-ingest scheduler: %s", exc)

    logger.info("Application startup complete (auth=%s)", AUTH_ENABLED)
    yield  # ── app runs ──

    # Shutdown
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
    logger.info("Application shutdown complete.")


# ── App instance ─────────────────────────────────────────────────
app = FastAPI(
    title="Corporate Law Assistant",
    version="2.0.0",
    description=(
        "AI-powered legal research assistant for Indian corporate law. "
        "Ask questions and get answers with cited sources from authoritative "
        "legal documents."
    ),
    lifespan=lifespan,
)

# ── CORS middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)

# ── Serve the frontend UI ───────────────────────────────────────
UI_DIR = PROJECT_ROOT / "ui"
if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


# ═════════════════════════════════════════════════════════════════
#  PUBLIC ENDPOINTS
# ═════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check for Render / monitoring."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Redirect bare root to the chat UI."""
    return RedirectResponse(url="/ui/index.html")


@app.get("/api/auth/config", response_model=AuthConfigResponse)
async def auth_config():
    """Return Supabase public config for the frontend.

    If SUPABASE_URL is empty, the frontend knows auth is disabled
    and will skip the login flow.
    """
    return AuthConfigResponse(
        supabase_url=SUPABASE_URL,
        supabase_anon_key=SUPABASE_ANON_KEY,
    )


@app.get("/docs/{filename}")
async def serve_pdf(filename: str):
    """Serve a legal PDF from data/documents/.

    Supports browser page navigation via #page=N fragment
    (handled entirely by the browser — the server just returns the PDF).

    Security:
      - Only files ending in .pdf are served
      - Path traversal is blocked (filename is stripped to basename)
      - PDFs are served inline (open in browser, not download)
    """
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are served")

    pdf_path = DOCUMENTS_DIR / safe_name
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Document not found: {safe_name}"
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=safe_name,
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


# ═════════════════════════════════════════════════════════════════
#  AUTHENTICATED ENDPOINTS
# ═════════════════════════════════════════════════════════════════

@app.post("/ask", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    user: dict = Depends(get_current_user),
):
    """Answer a legal question using retrieved PDF context."""
    start = time.time()
    user_id = user.get("sub", "anonymous")

    try:
        from src.rag_chain import ask
        result = ask(body.question)

        elapsed_ms = (time.time() - start) * 1000

        # Track analytics (non-blocking — failures don't affect response)
        track_question(user_id, body.question, elapsed_ms)
        track_activity(user_id, "ask")

        return AskResponse(**result)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "FAISS index not found. "
                "The knowledge base has not been indexed yet."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Error during /ask")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {exc}",
        ) from exc


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def analytics_endpoint(
    user: dict = Depends(get_current_user),
):
    """Return aggregated usage analytics."""
    track_activity(user.get("sub", "anonymous"), "view_analytics")
    data = get_analytics()
    return AnalyticsResponse(**data)


# ═════════════════════════════════════════════════════════════════
#  ADMIN ENDPOINTS (not exposed in UI — for admin use only)
# ═════════════════════════════════════════════════════════════════

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    user: dict = Depends(get_current_user),
):
    """Rebuild the FAISS index from all PDFs in data/documents/."""
    try:
        from src.vectorstore import create_vectorstore
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


@app.post("/auto-ingest", response_model=AutoIngestResponse)
async def auto_ingest_documents(
    user: dict = Depends(get_current_user),
):
    """Run the incremental auto-ingest pipeline."""
    try:
        from auto_ingest import run_pipeline

        stats = run_pipeline(skip_scrape=False)
        return AutoIngestResponse(
            status="success",
            new_chunks=stats["new_chunks"],
            reused_chunks=stats["reused_chunks"],
            pruned_chunks=stats["pruned_chunks"],
            total_vectors=stats["total_vectors"],
            elapsed_seconds=stats["elapsed_seconds"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No PDFs found: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Error during /auto-ingest")
        raise HTTPException(
            status_code=500,
            detail=f"Auto-ingestion failed: {exc}",
        ) from exc