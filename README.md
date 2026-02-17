# Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) service that answers questions over legal PDFs with **inline source citations**.  Built with FastAPI, LangChain 0.2.x, and FAISS — designed for local development with Ollama and production deployment with API-based LLMs.  Includes a built-in dark-themed chat UI served at `/ui`.

---

## Project Overview

**Problem:** Legal professionals spend significant time manually searching through large legal documents (e.g., the Companies Act — 400+ pages) to find relevant provisions. Traditional keyword search lacks semantic understanding and doesn't synthesise answers across sections.

**Solution:** This service ingests legal PDFs, splits them into semantically meaningful chunks, embeds them into a FAISS vector store, and retrieves the most relevant passages to answer user questions. A language model generates a structured answer with **traceable citations** (filename + page number), ensuring every claim is grounded in the source material.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE (one-time)                │
│                                                                  │
│  PDF files ──▶ PyPDFLoader ──▶ RecursiveCharacterTextSplitter    │
│                                  (1000 chars, 200 overlap)       │
│             ──▶ HuggingFaceEmbeddings ──▶ FAISS index (disk)     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE (per request)                │
│                                                                  │
│  User question ──▶ Embed query ──▶ FAISS retriever (top-k=3)     │
│                ──▶ Format context with source markers             │
│                ──▶ Legal prompt template ──▶ LLM ──▶ Answer       │
│                                                                  │
│  Response: { answer (with inline citations), sources [] }        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Framework** | FastAPI | Async-ready, auto-generated OpenAPI docs, Pydantic validation |
| **Orchestration** | LangChain 0.2.x | Mature RAG primitives (loaders, splitters, retrievers, prompts) |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | Lightweight (80 MB), runs on CPU, strong semantic similarity for legal text |
| **Vector Store** | FAISS (faiss-cpu) | Fast approximate nearest-neighbour search, zero infrastructure, local disk persistence |
| **LLM (local)** | Mistral via Ollama | Free, runs locally, good instruction-following for RAG |
| **LLM (prod)** | Groq / OpenAI / Together | API-based, no GPU needed on server, configurable via env vars |
| **PDF Parsing** | PyPDF | Reliable text extraction with page-level metadata |

---

## Data Ingestion Flow

```
python scripts/ingest.py
```

1. Scans `data/documents/` for all `.pdf` files
2. Loads each PDF page as a LangChain `Document` with metadata (`source`, `page`)
3. Splits into chunks (1000 chars, 200 overlap) preserving metadata
4. Embeds all chunks using `all-MiniLM-L6-v2`
5. Builds a FAISS index and saves to `data/vectorstore/`

**Re-ingestion:** The script performs a **full rebuild** every time — the entire index is replaced, not appended. This guarantees no duplicate chunks, since FAISS has no native upsert/deduplication. Safe to re-run at any time.

---

## RAG Query Flow

1. **Retrieve** — User question is embedded and matched against the FAISS index (top-3 chunks)
2. **Format** — Each chunk is labelled with `[Source: filename, Page N]` markers
3. **Generate** — A legal-domain prompt instructs the LLM to:
   - Answer using **only** the provided context
   - Include inline citations for every factual claim
   - Respond with a fixed "not enough information" message if the context is insufficient
4. **Return** — Structured JSON with the answer and a deduplicated source list

---

## API Endpoints

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `POST /ask`
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What powers do company directors have?"}'
```
```json
{
  "answer": "According to the Companies Act, directors have the following powers:\n\n1. Power to manage the affairs of the company [Source: companies_act.pdf, Page 142]\n2. Power to delegate authority to committees [Source: companies_act.pdf, Page 145]",
  "sources": [
    {
      "file": "companies_act.pdf",
      "page": 142,
      "snippet": "The Board of Directors of a company shall be entitled to exercise all such powers..."
    },
    {
      "file": "companies_act.pdf",
      "page": 145,
      "snippet": "Subject to the provisions of this Act, the Board may delegate..."
    }
  ]
}
```

### `POST /ingest`
Rebuilds the FAISS index from all PDFs in `data/documents/`.
```bash
curl -X POST http://localhost:8000/ingest
```
```json
{"status": "success", "chunks_indexed": 1678}
```

---

## Local Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd rag_legal_assistant

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place PDFs in data/documents/
# (companies_act.pdf is included by default)

# 5. Build the FAISS index
python scripts/ingest.py

# 6. Start Ollama (if using local LLM)
ollama serve
ollama pull mistral

# 7. Start the API server
uvicorn app.main:app --reload --port 8000

# 8. Open the chat UI in your browser
#    http://localhost:8000/ui
#    (or http://localhost:8000 — auto-redirects to the UI)

# 9. Test the API directly
curl http://localhost:8000/health
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama`, `openai`, `groq`, or `together` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (local dev only) |
| `OLLAMA_MODEL` | `mistral` | Model name for Ollama |
| `LLM_API_KEY` | — | API key for OpenAI / Groq / Together |
| `LLM_API_BASE` | auto-detected | Override API base URL if needed |
| `LLM_MODEL` | `mistral` | Model name at the API provider |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic answers |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_K` | `5` | Number of chunks retrieved per query |

**Switching providers (example — Groq):**
```bash
LLM_PROVIDER=groq
LLM_API_KEY=gsk_your_key_here
LLM_MODEL=mixtral-8x7b-32768
```

---

## Deployment (Render)

A `render.yaml` blueprint is included for one-click deployment.

**Startup command:**
```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Critical constraint:** Ollama cannot run on Render (no GPU, no persistent background process). You **must** use an API-based LLM provider in production:
- Set `LLM_PROVIDER=groq` (or `openai` / `together`)
- Set `LLM_API_KEY` in Render's environment variables dashboard

**FAISS persistence:** Render's free tier uses ephemeral disk — the index is lost on redeploy. Options:
1. Bundle the pre-built index in the repo (if small enough)
2. Call `POST /ingest` after each deploy
3. Upgrade to a Render plan with persistent disk

---

## Limitations

- **Single-document focus** — optimised for the Companies Act; not tested across diverse legal corpora
- **No authentication** — API endpoints are publicly accessible (add API key middleware for production)
- **Synchronous LLM calls** — `/ask` blocks until the LLM responds; no streaming support
- **No incremental ingestion** — full index rebuild required when adding new documents
- **Embedding model is English-only** — `all-MiniLM-L6-v2` has limited multilingual capability
- **No chunk-level caching** — repeated questions re-run the full retrieval + LLM pipeline

---

## Future Improvements

- [ ] **Streaming responses** — SSE/WebSocket support for real-time token output
- [ ] **Authentication** — API key or OAuth2 middleware
- [ ] **Multi-document upload** — REST endpoint to upload PDFs and trigger selective re-ingestion
- [ ] **Hybrid search** — combine FAISS (semantic) with BM25 (keyword) for better recall
- [ ] **Conversation memory** — multi-turn follow-up questions with chat history
- [ ] **Observability** — LangSmith or custom logging for retrieval quality monitoring
- [ ] **Evaluation framework** — automated answer quality scoring against a labelled test set

---

## Project Structure

```
rag_legal_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server (endpoints + lifespan + static UI)
│   └── schemas.py            # Pydantic request/response models
├── src/
│   ├── config.py             # Centralised env-var-driven settings
│   ├── loader.py             # PDF loading + chunking with metadata
│   ├── embeddings.py         # Singleton HuggingFace embedding model
│   ├── vectorstore.py        # FAISS create / load / persist (singleton)
│   ├── llm.py                # Multi-provider LLM factory
│   └── rag_chain.py          # RAG pipeline: retrieve → prompt → answer
├── ui/
│   ├── index.html            # Chat interface (dark theme)
│   ├── style.css             # Styling (Perplexity / ChatGPT aesthetic)
│   └── script.js             # Frontend logic (fetch → /ask)
├── scripts/
│   └── ingest.py             # CLI for building the FAISS index
├── data/
│   ├── documents/            # Source PDFs
│   └── vectorstore/          # Persisted FAISS index
├── requirements.txt
├── .env.example
├── render.yaml
└── README.md
```
