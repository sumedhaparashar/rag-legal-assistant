"""
PDF loading and chunking with legal metadata extraction.

Responsibilities
- Load one or many PDFs via PyPDFLoader.
- Split into chunks with RecursiveCharacterTextSplitter.
- Attach legal metadata (act_name, year, section, page, source_file) to
  every chunk so citations look like "Companies Act 2013, Section 166, p.42".
- Optionally rename PDFs on disk to their detected act name.

The function accepts an explicit path so it stays fully decoupled from
config — the caller decides which path to use.
"""

import os
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, ENABLE_AUTO_RENAME
from src.metadata_extractor import extract_metadata, clean_filename_for_act


def _load_single_pdf(pdf_path: Path):
    """Load one PDF and return (docs, first_page_text)."""
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    first_page_text = docs[0].page_content if docs else ""
    return docs, first_page_text


def _maybe_rename(pdf_path: Path, act_name: str) -> Path:
    """Rename the PDF to its detected act name (once). Returns new path."""
    if not ENABLE_AUTO_RENAME:
        return pdf_path
    if act_name in ("Unknown Act", ""):
        return pdf_path

    new_name = clean_filename_for_act(act_name)
    new_path = pdf_path.parent / new_name

    if new_path == pdf_path or new_path.exists():
        return pdf_path

    try:
        pdf_path.rename(new_path)
        print(f"  [RENAME] {pdf_path.name} → {new_name}")
        return new_path
    except OSError as e:
        print(f"  [WARN] Could not rename {pdf_path.name}: {e}")
        return pdf_path


def load_and_split(
    source: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List:
    """Load PDFs from *source* and return chunked Documents.

    Every chunk carries metadata:
      - act_name    : e.g. "Companies Act 2013"
      - year        : e.g. 2013
      - section     : e.g. "166"  (if detected)
      - page        : 0-based page index
      - source_file : original PDF filename
    """
    source = Path(source)

    if source.is_file():
        pdf_paths = [source]
    elif source.is_dir():
        pdf_paths = sorted(source.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {source}")
    else:
        raise FileNotFoundError(f"Source path does not exist: {source}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []

    for pdf_path in pdf_paths:
        print(f"[LOAD] {pdf_path.name}")
        docs, first_page_text = _load_single_pdf(pdf_path)

        # Detect act name on the raw PDF path (before rename)
        probe_meta = extract_metadata(first_page_text, "", pdf_path.name)
        act_name = probe_meta["act_name"]

        # Optionally rename file on disk
        pdf_path = _maybe_rename(pdf_path, act_name)
        source_file = pdf_path.name

        chunks = splitter.split_documents(docs)
        for ch in chunks:
            page_text = ch.page_content
            meta = extract_metadata(first_page_text, page_text, source_file)
            ch.metadata.update(meta)
            # preserve page number PyPDFLoader already set
            ch.metadata.setdefault("page", 0)

        all_chunks.extend(chunks)

    return all_chunks


# ── quick smoke-test ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import DOCUMENTS_DIR

    docs = load_and_split(DOCUMENTS_DIR)
    print(f"\nTotal chunks created: {len(docs)}")
    if docs:
        sample = docs[0]
        print(f"Sample metadata: {sample.metadata}")