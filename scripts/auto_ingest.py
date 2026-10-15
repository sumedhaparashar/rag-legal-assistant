"""
Automated document ingestion — scrape, hash, embed, index.

TRUE incremental pipeline:
  1. (Optional) Scrape a legal website for new PDF links and download them.
  2. Load & chunk all PDFs from data/documents/ (metadata is now legal-aware).
  3. SHA-256 each chunk. Skip chunks already in the registry.
  4. Embed ONLY new chunks. Add them to the existing FAISS index.
  5. Prune stale hashes for deleted documents.

Usage:
    python scripts/auto_ingest.py
    python scripts/auto_ingest.py --url https://example.gov
    python scripts/auto_ingest.py --skip-scrape
    python scripts/auto_ingest.py --delete file.pdf
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))          # so `from src.x import y` works
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # so `from x import y` also works


from config import (                                         # noqa: E402
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    HASH_REGISTRY_PATH,
    SCRAPE_URL,
    DEFAULT_ORG_ID,
)
from embeddings import get_embedding_model                   # noqa: E402
from hash_registry import HashRegistry                       # noqa: E402
from loader import load_and_split                            # noqa: E402
from vectorstore import add_documents_incremental            # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#   SCRAPING
# ═══════════════════════════════════════════════════════════════════

def scrape_pdf_links(url: str) -> List[str]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("[ERROR] 'requests' and 'beautifulsoup4' are required for scraping.")
        sys.exit(1)

    print(f"[SCRAPE] Fetching {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.lower().endswith(".pdf"):
            pdf_links.append(urljoin(url, href))

    pdf_links = list(dict.fromkeys(pdf_links))
    print(f"[SCRAPE] Found {len(pdf_links)} PDF link(s)")
    return pdf_links


def download_pdfs(pdf_urls: List[str], dest_dir: Path) -> List[Path]:
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
#   TRUE INCREMENTAL INGEST
# ═══════════════════════════════════════════════════════════════════

def incremental_ingest(
    source_dir: Path = DOCUMENTS_DIR,
    registry_path: Path = HASH_REGISTRY_PATH,
    persist_dir: Path = VECTORSTORE_DIR,
    org_id: str = DEFAULT_ORG_ID,
) -> Dict:
    start = time.perf_counter()

    # 1. Load and chunk (metadata is now legal-aware)
    print("[LOAD]  Loading and chunking PDFs …")
    chunks = load_and_split(source_dir)
    print(f"[LOAD]  {len(chunks)} chunk(s) from {source_dir}")

    # 2. Hash registry
    registry = HashRegistry(registry_path).load()
    print(f"[HASH]  Registry loaded — {registry.size} cached entries")

    # 3. Split chunks into new vs cached
    new_chunks = []
    active_hashes = set()
    now_iso = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        h = HashRegistry.compute_hash(chunk.page_content)
        active_hashes.add(h)

        if registry.contains(h):
            continue

        # tag with org_id and timestamp
        chunk.metadata["org_id"] = org_id
        chunk.metadata["last_updated"] = now_iso
        new_chunks.append((h, chunk))

    reused_count = len(chunks) - len(new_chunks)
    print(f"[EMBED] {len(new_chunks)} new chunk(s), {reused_count} reused")

    # 4. Add ONLY new chunks to the FAISS index
    added = 0
    if new_chunks:
        new_docs = [chunk for _, chunk in new_chunks]
        _, added = add_documents_incremental(new_docs, persist_dir=persist_dir)

        # Register in hash registry (no embeddings stored)
        for h, chunk in new_chunks:
            registry.put(
                h,
                metadata=chunk.metadata,
                text_preview=chunk.page_content,
            )

    # 5. Prune stale hashes
    pruned = registry.prune(active_hashes)
    if pruned:
        print(f"[PRUNE] Removed {pruned} stale hash(es)")

    registry.save()

    elapsed = time.perf_counter() - start
    stats = {
        "new_chunks": added,
        "reused_chunks": reused_count,
        "pruned_chunks": pruned,
        "elapsed_seconds": round(elapsed, 1),
    }

    print()
    print("[OK] Auto-ingest complete")
    print(f"     New chunks embedded : {stats['new_chunks']}")
    print(f"     Cached chunks reused: {stats['reused_chunks']}")
    print(f"     Stale chunks pruned : {stats['pruned_chunks']}")
    print(f"     Time elapsed        : {stats['elapsed_seconds']}s")
    print(f"     Index saved to      : {persist_dir}")

    return stats


# ═══════════════════════════════════════════════════════════════════
#   FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(url: Optional[str] = None, skip_scrape: bool = False,
                 delete_file: Optional[str] = None) -> Dict:

    if delete_file:
        target = DOCUMENTS_DIR / delete_file
        if target.exists():
            target.unlink()
            print(f"[DELETE] Removed {target}")
        else:
            print(f"[WARN]  File not found: {target}")

    if not skip_scrape:
        scrape_target = url or SCRAPE_URL
        if not scrape_target:
            print("[WARN] No SCRAPE_URL configured — running local-only.")
        else:
            pdf_urls = scrape_pdf_links(scrape_target)
            if pdf_urls:
                new_files = download_pdfs(pdf_urls, DOCUMENTS_DIR)
                print(f"[SCRAPE] {len(new_files)} new PDF(s) downloaded\n")

    return incremental_ingest()


def main():
    parser = argparse.ArgumentParser(
        description="Auto-ingest legal PDFs: scrape, hash, embed, index."
    )
    parser.add_argument("--url", default=None)
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--delete", default=None, metavar="FILENAME")
    args = parser.parse_args()

    run_pipeline(
        url=args.url,
        skip_scrape=args.skip_scrape,
        delete_file=args.delete,
    )


if __name__ == "__main__":
    main()