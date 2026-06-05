# ⚖️ RAG Legal Assistant

A **production‑ready Retrieval‑Augmented Generation (RAG) Legal Assistant** built with **FastAPI, LangChain, FAISS**, and **multi‑provider LLM support** (Groq / OpenAI / Together / Ollama).

The system answers legal questions **strictly based on provided documents**, returns **verifiable citations (file + page)**, and is designed to be **interview‑defensible and cloud‑deployable**.

---
<img width="1365" height="573" alt="WhatsApp Image 2026-04-23 at 11 20 20 PM (1)" src="https://github.com/user-attachments/assets/c95ae9a1-0d20-4b3c-b37a-8d67d2bccd36" />
<img width="1600" height="664" alt="WhatsApp Image 2026-04-23 at 11 20 20 PM" src="https://github.com/user-attachments/assets/d2102230-5114-4683-aad2-25868b38007c" />


## 🚀 Features

* 📚 **Document‑grounded answers** (no free hallucination)
* 🔎 **FAISS vector search** for millisecond‑level retrieval
* 🧠 **Multi‑LLM support** via a single environment variable
* 🏷️ **Explicit citations**: `[Source: filename, Page N]`
* ⚡ **Pre‑built vector index** (instant startup on deployment)
* 🌐 **FastAPI backend** with OpenAPI docs
* 💬 **ChatGPT‑style frontend** (pure HTML/CSS/JS, no build step)
* 🧪 **CLI ingestion script** for rebuilding the index

---
For Implementation Demo of the project : https://youtu.be/gG-0inAdGJA
## 🧱 Tech Stack

| Layer         | Technology                               | Reason                              |
| ------------- | ---------------------------------------- | ----------------------------------- |
| Language      | Python 3.11                              | Best ecosystem support for AI/RAG   |
| LLM Framework | LangChain 0.2.x                          | Prompting, loaders, retrievers      |
| LLM Providers | Groq / OpenAI / Together / Ollama        | Switchable via env var              |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight, CPU‑friendly           |
| Vector DB     | FAISS (CPU)                              | Fast, file‑based similarity search  |
| PDF Parsing   | PyPDF (`pypdf`)                          | Reliable text + metadata extraction |
| API Server    | FastAPI + Uvicorn                        | Async, production‑ready             |
| Frontend      | Vanilla HTML/CSS/JS                      | Zero build step                     |
| Deployment    | Render                                   | Simple Git‑based deploy             |

---

## 🏗️ Architecture Overview

```
User
  ↓
Frontend (HTML / JS)
  ↓ POST /ask
FastAPI (app/main.py)
  ↓
RAG Chain (src/rag_chain.py)
  ├─ Retrieve → FAISS
  ├─ Augment → Prompt + Context
  └─ Generate → LLM
  ↓
Answer + Sources
```

### RAG Flow

1. **User question** is embedded
2. **Top‑K relevant chunks** retrieved from FAISS
3. **Prompt augmented** with document context + strict instructions
4. **LLM generates answer** with mandatory citations
5. **Sources extracted & returned** to UI

---

## 📁 Project Structure

```
rag_legal_assistant/
├── app/                    # FastAPI backend (HTTP layer)
│   ├── main.py             # Routes, lifespan, static files
│   └── schemas.py          # Request/response models
│
├── src/                    # Core RAG logic
│   ├── config.py           # Centralised configuration
│   ├── loader.py           # PDF loading & chunking
│   ├── embeddings.py       # Embedding model singleton
│   ├── vectorstore.py      # FAISS create/load logic
│   ├── llm.py              # LLM provider factory
│   └── rag_chain.py        # RAG pipeline orchestrator
│
├── ui/                     # Frontend (served by FastAPI)
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── data/
│   ├── documents/          # Source legal PDFs
│   └── vectorstore/        # Pre‑built FAISS index
│
├── scripts/
│   └── ingest.py           # CLI index builder
│
├── requirements.txt
├── render.yaml             # Deployment blueprint
├── .env.example            # Environment variable template
└── README.md
```

---

## ⚙️ Configuration

All configuration is handled via **environment variables** (12‑factor style).

Example `.env`:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=your_api_key_here
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RETRIEVER_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

> ⚠️ **Important:** If you change the embedding model, you **must rebuild the FAISS index**.

---

## ▶️ Running Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Build / rebuild the vector index

```bash
python scripts/ingest.py
```

(Optional: ingest a single PDF)

```bash
python scripts/ingest.py path/to/file.pdf
```

### 3️⃣ Start the server

```bash
uvicorn app.main:app --reload
```

### 4️⃣ Open the UI

```
http://localhost:8000/ui
```

---

## 🔌 API Endpoints

| Method | Endpoint  | Description          |
| ------ | --------- | -------------------- |
| POST   | `/ask`    | Ask a legal question |
| POST   | `/ingest` | Rebuild FAISS index  |
| GET    | `/health` | Health check         |
| GET    | `/ui`     | Web chat interface   |

---

## 🧪 Example Request

```json
POST /ask
{
  "question": "What powers do company directors have?"
}
```

### Example Response

```json
{
  "answer": "Directors have the power to manage the affairs of the company... [Source: companies_act.pdf, Page 42]",
  "sources": [
    {
      "file": "companies_act.pdf",
      "page": 42,
      "snippet": "The Board of Directors shall be entitled to exercise all such powers..."
    }
  ]
}
```

---

## 🎯 Key Design Decisions

* **Pre‑built FAISS index committed to git** → instant startup on Render
* **Full rebuild on ingest** → avoids FAISS duplication issues
* **Singleton embeddings & vectorstore** → heavy models loaded once
* **OpenAI‑compatible APIs** → one LangChain class for Groq/OpenAI/Together
* **Prescriptive prompt** → forces citations & prevents hallucination
* **UI served by FastAPI** → no CORS issues, single deployment

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**.
Responses are AI‑generated and **do not constitute legal advice**.

---


## ⭐ If you like this project

Give it a ⭐ and feel free to fork or extend it 🚀
