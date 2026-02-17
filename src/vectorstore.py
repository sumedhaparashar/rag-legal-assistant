"""
FAISS vector store manager.

Provides two operations:
  1. create_vectorstore()  — build a new index from documents and persist it.
  2. load_vectorstore()    — load the persisted index (singleton).

Design decision:  The loaded store is cached in a module-level variable
so the (relatively expensive) disk-read + deserialization happens only
once per process lifetime.  This is safe because FAISS is read-only at
query time.
"""

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS

from config import VECTORSTORE_DIR, DOCUMENTS_DIR
from embeddings import get_embedding_model
from loader import load_and_split

_vectorstore = None


def create_vectorstore(
    source=None,
    persist_dir=None,
) -> FAISS:
    """Ingest PDFs → embed → build FAISS index → save to disk.

    Parameters
    ----------
    source : str | Path, optional
        PDF file or directory.  Defaults to ``config.DOCUMENTS_DIR``.
    persist_dir : str | Path, optional
        Where to save the index.  Defaults to ``config.VECTORSTORE_DIR``.
    """
    source = Path(source) if source else DOCUMENTS_DIR
    persist_dir = Path(persist_dir) if persist_dir else VECTORSTORE_DIR

    docs = load_and_split(source)
    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(docs, embeddings)

    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir))

    # Also update the singleton so subsequent queries use the fresh index
    global _vectorstore
    _vectorstore = vectorstore

    return vectorstore


def load_vectorstore(persist_dir=None) -> FAISS:
    """Load a previously-persisted FAISS index (singleton).

    Returns the cached instance on subsequent calls.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    persist_dir = Path(persist_dir) if persist_dir else VECTORSTORE_DIR

    if not (persist_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index found at {persist_dir}.  "
            "Run create_vectorstore() first."
        )

    embeddings = get_embedding_model()
    _vectorstore = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return _vectorstore


# ── quick smoke-test ─────────────────────────────────────────────
if __name__ == "__main__":
    vs = create_vectorstore()
    print(f"FAISS index created — {vs.index.ntotal} vectors")
