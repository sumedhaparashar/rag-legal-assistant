# ⚖️ Corporate Law Assistant

**AI-powered legal research assistant** for Indian corporate law. Ask questions about company regulations, director duties, SEBI guidelines, banking laws, and more — with every answer backed by cited sources from authoritative legal documents.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## ✨ Features

| Feature | Description |
|---|---|
| **RAG Pipeline** | Retrieval-Augmented Generation using FAISS vector search + LLM |
| **Cited Sources** | Every answer includes document name, page number, and relevant snippet |
| **Multi-Provider LLM** | Supports Groq, OpenAI, Together AI, and local Ollama |
| **Authentication** | Supabase Auth with Google Sign-In and email/password |
| **Analytics Dashboard** | Track usage metrics, popular topics, response times |
| **Auto-Ingest** | Incremental document ingestion with hash-based deduplication |
| **Production-Ready** | Docker deployment, health checks, structured logging |

## 🏗️ Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│   Frontend   │─────▶│   FastAPI + Auth  │─────▶│  Groq LLM    │
│  (HTML/JS)   │      │   (uvicorn)       │      │  (Llama 3.3) │
└──────────────┘      └────────┬─────────┘      └──────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
              ┌─────▼─────┐        ┌─────▼──────┐
              │   FAISS    │        │  Analytics  │
              │ VectorStore│        │  (SQLite)   │
              └─────┬──────┘        └────────────┘
                    │
              ┌─────▼──────┐
              │ HuggingFace│
              │ Embeddings │
              │(MiniLM-L6) │
              └────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vanilla HTML/CSS/JS, Supabase JS SDK |
| **Backend** | Python 3.11, FastAPI, uvicorn |
| **LLM** | Groq API (Llama 3.3 70B) via LangChain |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| **Vector Store** | FAISS (CPU) with cosine/euclidean/hybrid search |
| **Auth** | Supabase Auth (JWT verification) |
| **Analytics** | SQLite |
| **Deployment** | Docker on Render |
| **PDF Processing** | PyPDF + LangChain text splitters |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com) (free tier available)
- (Optional) [Supabase project](https://supabase.com) for authentication

### Local Development

```bash
# Clone the repository
git clone https://github.com/sumedhaparashar/rag-legal-assistant.git
cd rag-legal-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Docker

```bash
docker build -t corporate-law-assistant .
docker run -p 8000:8000 --env-file .env corporate-law-assistant
```

## 📡 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/` | No | Redirect to chat UI |
| `GET` | `/health` | No | Health check |
| `GET` | `/api/auth/config` | No | Supabase public configuration |
| `POST` | `/ask` | Yes | Ask a legal question |
| `GET` | `/api/analytics` | Yes | Usage analytics |
| `POST` | `/ingest` | Yes | Rebuild FAISS index (admin) |
| `POST` | `/auto-ingest` | Yes | Incremental re-index (admin) |

### Example: Ask a Question

```bash
curl -X POST https://your-app.onrender.com/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"question": "What are the duties of company directors?"}'
```

Response:
```json
{
  "answer": "Company directors have several key duties...",
  "sources": [
    {
      "file": "companies_act.pdf",
      "page": 42,
      "snippet": "Section 166 — Duties of directors...",
      "relevance_score": 0.87
    }
  ]
}
```

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | LLM provider: `groq`, `openai`, `together`, `ollama` |
| `LLM_API_KEY` | Yes* | API key for the LLM provider |
| `LLM_MODEL` | No | Model name (default: `llama-3.3-70b-versatile`) |
| `SUPABASE_URL` | No | Supabase project URL (leave blank to disable auth) |
| `SUPABASE_ANON_KEY` | No | Supabase anonymous/public key |
| `SUPABASE_JWT_SECRET` | No | Supabase JWT secret for backend verification |
| `EMBEDDING_MODEL_NAME` | No | HuggingFace model (default: `all-MiniLM-L6-v2`) |
| `RETRIEVER_K` | No | Number of documents to retrieve (default: `5`) |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins |

*Not required when `LLM_PROVIDER=ollama`

## 📂 Project Structure

```
rag_legal_assistant/
├── app/
│   ├── main.py          # FastAPI application + endpoint definitions
│   ├── schemas.py        # Pydantic request/response models
│   ├── auth.py           # Supabase JWT authentication middleware
│   └── analytics.py      # SQLite analytics tracking
├── src/
│   ├── config.py         # Centralised configuration (env vars)
│   ├── rag_chain.py      # RAG pipeline (retrieve + generate + cite)
│   ├── vectorstore.py    # FAISS vector store manager
│   ├── embeddings.py     # HuggingFace embedding model factory
│   ├── llm.py            # Multi-provider LLM factory
│   ├── loader.py         # PDF loading and chunking
│   └── hash_registry.py  # Content-based deduplication for ingestion
├── ui/
│   ├── index.html        # Chat interface
│   ├── auth.html         # Login / signup page
│   ├── analytics.html    # Analytics dashboard
│   ├── style.css         # Design system (dark theme)
│   └── script.js         # Frontend application logic
├── data/
│   ├── documents/        # Legal PDF corpus (12 documents)
│   └── vectorstore/      # Pre-built FAISS index
├── scripts/
│   ├── auto_ingest.py    # Automated incremental ingestion
│   └── ingest.py         # Manual full-rebuild ingestion
├── Dockerfile            # Production Docker image
├── render.yaml           # Render deployment blueprint
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

## 📚 Legal Document Corpus

The knowledge base includes 12 authoritative Indian legal documents:

- Companies Act, 2013
- Indian Penal Code, 1860
- Banking Regulation Act, 1949
- Competition Act, 2002
- Copyright Act, 1957
- Patents Act, 1970
- Foreign Contribution Regulation Act, 2010
- SEBI Annual Report & Regulations
- Producer Company Regulations

## 🔒 Security

- **Authentication**: JWT-based auth via Supabase (HS256)
- **Authorization**: Only administrators can modify the legal knowledge base
- **CORS**: Restricted to configured origins
- **Input validation**: Pydantic schemas with length limits
- **No user uploads**: The legal corpus is curated and managed by administrators only

## 📊 Analytics

The built-in analytics dashboard tracks:
- Total registered users and daily active users
- Total legal questions asked
- Average response time
- Most frequently asked legal topics
- Server uptime
- Daily question volume (7-day chart)

## 🚢 Deployment

This application is configured for deployment on **Render** using Docker:

1. Fork this repository
2. Connect your GitHub account to [Render](https://render.com)
3. Create a new Web Service → select this repo
4. Render auto-detects the `render.yaml` blueprint
5. Add your environment variables (LLM_API_KEY, Supabase keys) in Render's dashboard
6. Deploy

## License

MIT
