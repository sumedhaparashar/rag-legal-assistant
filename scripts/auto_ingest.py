"""
Automated document ingestion — scrape, hash, embed, index.

Replaces the manual full-rebuild workflow with an incremental pipeline:
  1. (Optional) Scrape a legal website for new PDF links and download them.
  2. Load & chunk all PDFs from data/documents/.
  3. Compute SHA-256 hashes; reuse cached embeddings where possible.
  4. Rebuild the FAISS index from the combined (cached + new) embeddings.
  5. Prune stale hashes for deleted documents.

Usage:
    python scripts/auto_ingest.py                            # scrape SCRAPE_URL + re-index
    python scripts/auto_ingest.py --url https://example.gov  # override scrape URL
    python scripts/auto_ingest.py --skip-scrape              # only re-index local PDFs
    python scripts/auto_ingest.py --delete file.pdf          # remove a specific PDF, then re-index
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

# ── Make src/ importable ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (                                         # noqa: E402
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    HASH_REGISTRY_PATH,
    SCRAPE_URL,
)
from embeddings import get_embedding_model                   # noqa: E402
from hash_registry import HashRegistry                       # noqa: E402
from loader import load_and_split                            # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#   STEP 1 — Scrape PDF links from a web page
# ═══════════════════════════════════════════════════════════════════

def scrape_pdf_links(url: str) -> List[str]:
    """Discover all PDF links on *url* and return their absolute URLs."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("[ERROR] 'requests' and 'beautifulsoup4' are required for scraping.")
        print("        Install them:  pip install requests beautifulsoup4")
        sys.exit(1)

    print(f"[SCRAPE] Fetching {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.lower().endswith(".pdf"):
            absolute = urljoin(url, href)
            pdf_links.append(absolute)

    pdf_links = list(dict.fromkeys(pdf_links))  # deduplicate, preserve order
    print(f"[SCRAPE] Found {len(pdf_links)} PDF link(s)")
    return pdf_links


def download_pdfs(pdf_urls: List[str], dest_dir: Path) -> List[Path]:
    """Download PDFs that don't already exist locally. Returns paths of new files."""
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    new_files = []

    for url in pdf_urls:
        filename = Path(urlparse(url).path).name
        if not filename:
            continue
        dest_path = dest_dir / filename

        if dest_path.exists():
            print(f"  [SKIP] {filename} (already exists)")
            continue

        print(f"  [DOWNLOAD] {filename}")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
            new_files.append(dest_path)
        except Exception as exc:
            print(f"  [WARN] Failed to download {filename}: {exc}")

    return new_files


# ═══════════════════════════════════════════════════════════════════
#   STEP 2–6 — Load, hash, embed, rebuild index
# ═══════════════════════════════════════════════════════════════════

def incremental_ingest(
    source_dir: Path = DOCUMENTS_DIR,
    registry_path: Path = HASH_REGISTRY_PATH,
    persist_dir: Path = VECTORSTORE_DIR,
) -> Dict:
    """Load PDFs, compute hashes, embed only new chunks, rebuild FAISS.

    Returns a stats dict: {new, reused, pruned, total, elapsed}.
    """
    from langchain_community.vectorstores import FAISS

    start = time.perf_counter()

    # ── Load & chunk ─────────────────────────────────────────────
    print("[LOAD]  Loading and chunking PDFs …")
    chunks = load_and_split(source_dir)
    print(f"[LOAD]  {len(chunks)} chunk(s) from {source_dir}")

    # ── Load hash registry ───────────────────────────────────────
    registry = HashRegistry(registry_path).load()
    print(f"[HASH]  Registry loaded — {registry.size} cached entries")

    # ── Embedding model ──────────────────────────────────────────
    embed_model = get_embedding_model()

    # ── Classify chunks as new or cached ─────────────────────────
    new_texts = []
    new_meta = []
    all_embeddings = []
    all_texts = []
    all_metadatas = []
    active_hashes = set()
    new_count = 0
    reused_count = 0

    now_iso = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        h = HashRegistry.compute_hash(chunk.page_content)
        active_hashes.add(h)

        cached = registry.get(h)
        if cached is not None:
            # Reuse cached embedding
            all_embeddings.append(cached["embedding"])
            meta = {**chunk.metadata, "last_updated": now_iso}
            all_texts.append(chunk.page_content)
            all_metadatas.append(meta)
            reused_count += 1
        else:
            # Queue for embedding
            new_texts.append(chunk.page_content)
            new_meta.append(chunk.metadata)

    # ── Embed new chunks in batch ────────────────────────────────
    if new_texts:
        print(f"[EMBED] Embedding {len(new_texts)} new chunk(s) …")
        new_embeddings = embed_model.embed_documents(new_texts)

        for text, emb, meta in zip(new_texts, new_embeddings, new_meta):
            h = HashRegistry.compute_hash(text)
            full_meta = {**meta, "last_updated": now_iso}
            registry.put(h, emb, full_meta, text_preview=text)
            all_embeddings.append(emb)
            all_texts.append(text)
            all_metadatas.append(full_meta)
            new_count += 1
    else:
        print("[EMBED] All chunks cached — 0 new embeddings needed")

    # ── Prune stale entries ──────────────────────────────────────
    pruned = registry.prune(active_hashes)
    if pruned:
        print(f"[PRUNE] Removed {pruned} stale hash(es)")

    # ── Build FAISS index ────────────────────────────────────────
    print("[INDEX] Building FAISS index …")
    text_embedding_pairs = list(zip(all_texts, all_embeddings))

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embed_model,
        metadatas=all_metadatas,
    )
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir))

    # ── Save registry ────────────────────────────────────────────
    registry.save()

    elapsed = time.perf_counter() - start

    stats = {
        "new_chunks": new_count,
        "reused_chunks": reused_count,
        "pruned_chunks": pruned,
        "total_vectors": vectorstore.index.ntotal,
        "elapsed_seconds": round(elapsed, 1),
    }

    print()
    print(f"[OK] Auto-ingest complete")
    print(f"     New chunks embedded : {stats['new_chunks']}")
    print(f"     Cached chunks reused: {stats['reused_chunks']}")
    print(f"     Stale chunks pruned : {stats['pruned_chunks']}")
    print(f"     Total vectors       : {stats['total_vectors']}")
    print(f"     Time elapsed        : {stats['elapsed_seconds']}s")
    print(f"     Index saved to      : {persist_dir}")

    return stats


# ═══════════════════════════════════════════════════════════════════
#   FULL PIPELINE (scrape + ingest)
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(
    url: Optional[str] = None,
    skip_scrape: bool = False,
    delete_file: Optional[str] = None,
) -> Dict:
    """End-to-end: scrape → download → ingest.  Returns stats dict."""

    # ── Optional deletion ────────────────────────────────────────
    if delete_file:
        target = DOCUMENTS_DIR / delete_file
        if target.exists():
            target.unlink()
            print(f"[DELETE] Removed {target}")
        else:
            print(f"[WARN]  File not found: {target}")

    # ── Optional scraping ────────────────────────────────────────
    if not skip_scrape:
        scrape_target = url or SCRAPE_URL
        if not scrape_target:
            print("[WARN] No SCRAPE_URL configured and --url not provided.")
            print("       Running in local-only mode (same as --skip-scrape).")
        else:
            pdf_urls = scrape_pdf_links(scrape_target)
            if pdf_urls:
                new_files = download_pdfs(pdf_urls, DOCUMENTS_DIR)
                print(f"[SCRAPE] {len(new_files)} new PDF(s) downloaded\n")

    # ── Incremental ingest ───────────────────────────────────────
    return incremental_ingest()


# ═══════════════════════════════════════════════════════════════════
#   CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Auto-ingest legal PDFs: scrape, hash, embed, index."
    )
    parser.add_argument(
        "--url",
        default=None,
        help="URL to scrape for PDF links (overrides SCRAPE_URL env var)",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip web scraping; only re-index local PDFs in data/documents/",
    )
    parser.add_argument(
        "--delete",
        default=None,
        metavar="FILENAME",
        help="Delete a specific PDF from data/documents/ before re-indexing",
    )
    args = parser.parse_args()

    run_pipeline(
        url=args.url,
        skip_scrape=args.skip_scrape,
        delete_file=args.delete,
    )


if __name__ == "__main__":
    main()
