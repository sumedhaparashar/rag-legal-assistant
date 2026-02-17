"""
Ingestion CLI — build (or rebuild) the FAISS vector index from PDFs.

Usage:
    python scripts/ingest.py                    # ingest all PDFs in data/documents/
    python scripts/ingest.py path/to/file.pdf   # ingest a single PDF

Re-ingestion safety
-------------------
This script always REPLACES the entire FAISS index rather than appending.
That means:
  • It is **safe to re-run** — you will never get duplicate chunks.
  • If you add new PDFs to data/documents/ and re-run, the old PDFs
    are re-indexed alongside the new ones (the whole directory is scanned).

Why full-rebuild instead of incremental?
  FAISS (flat / IVF) has no built-in deduplication or upsert.  Appending
  would risk duplicating chunks if the same PDF is ingested twice.  For
  the typical legal-document use case (tens to low hundreds of PDFs), a
  full rebuild takes seconds and is the simplest correct approach.
"""

import sys
import time
from pathlib import Path

# ── Make `src/` importable no matter where the script is invoked ─
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DOCUMENTS_DIR, VECTORSTORE_DIR          # noqa: E402
from vectorstore import create_vectorstore                  # noqa: E402


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCUMENTS_DIR

    print(f"[SOURCE]  {source}")
    print(f"[INDEX]   {VECTORSTORE_DIR}")
    print()

    start = time.perf_counter()
    vs = create_vectorstore(source=source, persist_dir=VECTORSTORE_DIR)
    elapsed = time.perf_counter() - start

    n_vectors = vs.index.ntotal
    print(f"[OK] Ingestion complete")
    print(f"     Vectors indexed : {n_vectors}")
    print(f"     Time elapsed    : {elapsed:.1f}s")
    print(f"     Index saved to  : {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
