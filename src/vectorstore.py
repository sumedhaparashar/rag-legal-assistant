"""
FAISS vector store manager with multiple similarity metrics
and TRUE incremental indexing.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import VECTORSTORE_DIR, DOCUMENTS_DIR, DEFAULT_ORG_ID
from src.embeddings import get_embedding_model
from src.loader import load_and_split

_vectorstore = None


# ═══════════════════════════════════════════════════════════════════
#   CREATE / LOAD
# ═══════════════════════════════════════════════════════════════════

def create_vectorstore(source=None, persist_dir=None) -> FAISS:
    """Full rebuild — embed everything from scratch. Used for first-time setup."""
    source = Path(source) if source else DOCUMENTS_DIR
    persist_dir = Path(persist_dir) if persist_dir else VECTORSTORE_DIR

    docs = load_and_split(source)
    embeddings = get_embedding_model()

    vectorstore = FAISS.from_documents(docs, embeddings)

    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir))

    global _vectorstore
    _vectorstore = vectorstore
    return vectorstore


def add_documents_incremental(
    new_docs: List[Document],
    persist_dir=None,
) -> Tuple[FAISS, int]:
    """Add ONLY new documents to the existing index. Returns (vs, added_count).

    This is the true incremental path. It does NOT rebuild the whole index.
    """
    global _vectorstore
    persist_dir = Path(persist_dir) if persist_dir else VECTORSTORE_DIR

    if not new_docs:
        return load_vectorstore(persist_dir), 0

    embeddings = get_embedding_model()

    # Load existing index, or create one if none exists
    try:
        vs = load_vectorstore(persist_dir)
    except FileNotFoundError:
        vs = FAISS.from_documents(new_docs, embeddings)
        persist_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(persist_dir))
        _vectorstore = vs
        return vs, len(new_docs)

    # Add new embeddings to existing index
    texts = [d.page_content for d in new_docs]
    metadatas = [d.metadata for d in new_docs]
    new_embeddings = embeddings.embed_documents(texts)

    vs.add_embeddings(
        text_embeddings=list(zip(texts, new_embeddings)),
        metadatas=metadatas,
    )

    vs.save_local(str(persist_dir))
    _vectorstore = vs
    return vs, len(new_docs)


def load_vectorstore(persist_dir=None) -> FAISS:
    """Load a persisted FAISS index (singleton)."""
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


# ═══════════════════════════════════════════════════════════════════
#   SEARCH — three metrics, with optional org_id filter
# ═══════════════════════════════════════════════════════════════════

def _filter_by_org(docs_with_scores, org_id: Optional[str]):
    """Drop chunks that don't belong to the given org (if org_id is set)."""
    if not org_id:
        return docs_with_scores
    out = []
    for doc, score in docs_with_scores:
        doc_org = doc.metadata.get("org_id", DEFAULT_ORG_ID)
        if doc_org == org_id:
            out.append((doc, score))
    return out


def similarity_search_cosine(query: str, k: int = 5,
                             score_threshold: Optional[float] = None,
                             org_id: Optional[str] = None
                             ) -> List[Tuple[Document, float]]:
    vs = load_vectorstore()

    # Over-fetch if we plan to filter by org
    fetch_k = k * 3 if org_id else k
    docs_with_scores = vs.similarity_search_with_score(query, fetch_k)

    cosine_results = []
    for doc, l2_distance in docs_with_scores:
        cosine_sim = 1 - (l2_distance ** 2) / 2
        cosine_sim = max(0.0, min(1.0, cosine_sim))
        if score_threshold is None or cosine_sim >= score_threshold:
            cosine_results.append((doc, cosine_sim))

    cosine_results = _filter_by_org(cosine_results, org_id)
    return cosine_results[:k]


def similarity_search_euclidean(query: str, k: int = 5,
                                normalize_scores: bool = True,
                                org_id: Optional[str] = None
                                ) -> List[Tuple[Document, float]]:
    vs = load_vectorstore()
    fetch_k = k * 3 if org_id else k
    docs_with_distances = vs.similarity_search_with_score(query, fetch_k)

    if not normalize_scores:
        docs_with_distances = _filter_by_org(docs_with_distances, org_id)
        return docs_with_distances[:k]

    similarity_results = []
    for doc, distance in docs_with_distances:
        similarity = 1.0 / (1.0 + distance)
        similarity_results.append((doc, similarity))

    similarity_results = _filter_by_org(similarity_results, org_id)
    return similarity_results[:k]


def hybrid_search(query: str, k: int = 5,
                  cosine_weight: float = 0.6,
                  euclidean_weight: float = 0.4,
                  normalize_scores: bool = True,
                  org_id: Optional[str] = None
                  ) -> List[Tuple[Document, float]]:
    candidate_k = k * 2

    cosine_results = similarity_search_cosine(query, k=candidate_k, org_id=org_id)
    euclidean_results = similarity_search_euclidean(
        query, k=candidate_k, normalize_scores=True, org_id=org_id
    )

    combined = {}
    for doc, score in cosine_results:
        key = doc.page_content
        combined[key] = {"document": doc, "score": score * cosine_weight}

    for doc, score in euclidean_results:
        key = doc.page_content
        if key in combined:
            combined[key]["score"] += score * euclidean_weight
        else:
            combined[key] = {"document": doc, "score": score * euclidean_weight}

    results = [(v["document"], v["score"]) for v in combined.values()]
    results.sort(key=lambda x: x[1], reverse=True)

    if normalize_scores and results:
        max_score = results[0][1]
        if max_score > 0:
            results = [(d, s / max_score) for d, s in results]

    return results[:k]


def similarity_search_with_rerank(query: str, k: int = 5,
                                  candidate_multiplier: int = 3,
                                  similarity_metric: str = "cosine",
                                  org_id: Optional[str] = None
                                  ) -> List[Tuple[Document, float]]:
    candidate_k = k * candidate_multiplier

    if similarity_metric == "cosine":
        cands = similarity_search_cosine(query, k=candidate_k, org_id=org_id)
    elif similarity_metric == "euclidean":
        cands = similarity_search_euclidean(query, k=candidate_k,
                                            normalize_scores=True, org_id=org_id)
    elif similarity_metric == "hybrid":
        cands = hybrid_search(query, k=candidate_k, org_id=org_id)
    else:
        raise ValueError(f"Unknown metric: {similarity_metric}")

    return cands[:k]


def similarity_search(query: str, k: int = 5, metric: str = "cosine",
                     org_id: Optional[str] = None) -> List[Document]:
    if metric == "cosine":
        results = similarity_search_cosine(query, k, org_id=org_id)
    elif metric == "euclidean":
        results = similarity_search_euclidean(query, k, org_id=org_id)
    elif metric == "hybrid":
        results = hybrid_search(query, k, org_id=org_id)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return [doc for doc, _ in results]


def get_vectorstore_stats() -> Dict:
    vs = load_vectorstore()
    stats = {
        "total_vectors": vs.index.ntotal,
        "dimension": vs.index.d,
        "index_type": type(vs.index).__name__,
        "persist_dir": str(VECTORSTORE_DIR),
    }
    if VECTORSTORE_DIR.exists():
        stats["index_files_exist"] = (
            (VECTORSTORE_DIR / "index.faiss").exists()
            and (VECTORSTORE_DIR / "index.pkl").exists()
        )
    else:
        stats["index_files_exist"] = False
    return stats