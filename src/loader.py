"""
PDF loading and chunking.

Responsibilities
- Load one or many PDFs via PyPDFLoader.
- Split into chunks with RecursiveCharacterTextSplitter.
- Preserve metadata (source filename, page number) on every chunk
  so downstream components can build citations.

Design decision:  The function accepts an explicit path (single file
or directory) instead of importing a hardcoded path, keeping this
module fully decoupled from config — the *caller* decides which
path to use.
"""

from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split(
    source: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List:
    """Load PDFs from *source* (file or directory) and return chunked Documents.

    Each returned Document carries metadata with at least:
      - ``source``  – original PDF file path
      - ``page``    – 0-based page index
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

    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(all_docs)
    return chunks


# ── quick smoke-test ─────────────────────────────────────────────
if __name__ == "__main__":
    from config import DOCUMENTS_DIR

    docs = load_and_split(DOCUMENTS_DIR)
    print(f"Total chunks created: {len(docs)}")
    if docs:
        sample = docs[0]
        print(f"Sample metadata: {sample.metadata}")
