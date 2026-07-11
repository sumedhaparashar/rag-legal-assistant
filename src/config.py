"""
Centralised configuration for the RAG Legal Assistant.

All tuneable parameters live here. Values are read from environment
variables (via os.environ) with sensible defaults so the app works
out-of-the-box on a dev machine without a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (local dev); on Render env vars come from the dashboard.
load_dotenv()

# ── Project root (two levels up from src/config.py) ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Document / vector store paths ────────────────────────────────
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# ── Auto-ingest / scraping ──────────────────────────────────────
SCRAPE_URL = os.getenv("SCRAPE_URL", "")
HASH_REGISTRY_PATH = PROJECT_ROOT / "data" / "hash_registry.json"
AUTO_INGEST_INTERVAL_HOURS = int(os.getenv("AUTO_INGEST_INTERVAL_HOURS", "0"))

# ── Embedding model ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)

# ── Text splitter ────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retriever ────────────────────────────────────────────────────
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))

# ── LLM provider selection ──────────────────────────────────────
# Supported values: "ollama", "openai", "groq", "together"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# -- Ollama (local) -----------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# -- OpenAI-compatible API (OpenAI / Groq / Together) -------------
#    All three use the same ChatOpenAI interface; only the base URL
#    and API key differ.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "")       # auto-set per provider if blank
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# -- Shared --------------------------------------------------------
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ── Supabase Auth ────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# ── Analytics ────────────────────────────────────────────────────
ANALYTICS_DB_PATH = PROJECT_ROOT / "data" / "analytics.db"

# ── Allowed CORS Origins ────────────────────────────────────────
_default_origins = (
    "http://localhost,http://localhost:8000,"
    "http://127.0.0.1,http://127.0.0.1:8000"
)
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]

# ── Logging ──────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()