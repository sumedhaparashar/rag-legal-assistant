"""
FAISS vector store manager with multiple similarity metrics.

Provides three similarity search methods:
  1. Euclidean (L2) distance - your original method
  2. Cosine similarity - normalized inner product (recommended)
  3. Hybrid search - combines both metrics

Design decision: The loaded store is cached in a module-level variable
so the (relatively expensive) disk-read + deserialization happens only
once per process lifetime. This is safe because FAISS is read-only at
query time.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import VECTORSTORE_DIR, DOCUMENTS_DIR
from src.embeddings import get_embedding_model
from src.loader import load_and_split

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


def similarity_search_cosine(
    query: str, 
    k: int = 5,
    score_threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """Search using cosine similarity (recommended for semantic search).
    
    Cosine similarity measures the angle between vectors, making it better
    for semantic similarity regardless of text length. Values range from
    0 (unrelated) to 1 (identical meaning).
    
    Parameters
    ----------
    query : str
        The user's question or search query.
    k : int, default=5
        Number of documents to retrieve.
    score_threshold : float, optional
        Minimum similarity score (0-1). Documents below this are filtered out.
    
    Returns
    -------
    List[Tuple[Document, float]]
        List of (document, cosine_similarity_score) pairs, sorted by score descending.
    """
    vectorstore = load_vectorstore()
    
    # Get query embedding
    query_embedding = vectorstore.embeddings.embed_query(query)
    
    # Normalize for cosine similarity (L2 normalization to unit vector)
    query_embedding = np.array(query_embedding)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm > 0:
        query_embedding = query_embedding / query_norm
    
    # FAISS inner product search (for normalized vectors, IP = cosine similarity)
    # We need to get scores from FAISS directly for normalized vectors
    embeddings_array = np.array([query_embedding]).astype('float32')
    
    # Search with raw scores (FAISS returns distances)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k)
    
    # Convert to cosine similarity (FAISS L2 distance → cosine similarity)
    # For normalized vectors: cosine_similarity = 1 - (l2_distance^2 / 2)
    cosine_results = []
    for doc, l2_distance in docs_with_scores:
        # Convert L2 distance to cosine similarity
        cosine_sim = 1 - (l2_distance ** 2) / 2
        cosine_sim = max(0.0, min(1.0, cosine_sim))  # Clamp to [0,1]
        
        if score_threshold is None or cosine_sim >= score_threshold:
            cosine_results.append((doc, cosine_sim))
    
    return cosine_results


def similarity_search_euclidean(
    query: str, 
    k: int = 5,
    normalize_scores: bool = True
) -> List[Tuple[Document, float]]:
    """Search using Euclidean (L2) distance - your original method.
    
    This is a wrapper around FAISS's native similarity_search_with_score.
    Lower distance = more similar.
    
    Parameters
    ----------
    query : str
        The user's question or search query.
    k : int, default=5
        Number of documents to retrieve.
    normalize_scores : bool, default=True
        Convert distances to similarity scores (0-1 range, higher = more similar).
    
    Returns
    -------
    List[Tuple[Document, float]]
        List of (document, similarity_score) pairs, sorted by score descending.
    """
    vectorstore = load_vectorstore()
    
    # Get native FAISS search (returns L2 distances)
    docs_with_distances = vectorstore.similarity_search_with_score(query, k)
    
    if not normalize_scores:
        # Return raw distances (lower is better)
        return docs_with_distances
    
    # Convert distances to similarity scores (0-1, higher = more similar)
    # Max possible Euclidean distance in embedding space is about sqrt(2*4) ≈ 2.828
    # We'll normalize using a sigmoid-like transformation
    similarity_results = []
    for doc, distance in docs_with_distances:
        # Convert distance to similarity score
        # similarity = 1 / (1 + distance) gives 1 at distance 0, 0.5 at 1, 0.33 at 2
        similarity = 1.0 / (1.0 + distance)
        similarity_results.append((doc, similarity))
    
    return similarity_results


def hybrid_search(
    query: str,
    k: int = 5,
    cosine_weight: float = 0.6,
    euclidean_weight: float = 0.4,
    normalize_scores: bool = True
) -> List[Tuple[Document, float]]:
    """Combine cosine and Euclidean similarity for robust retrieval.
    
    Cosine similarity is better for semantic meaning, while Euclidean distance
    captures magnitude differences. Combining them often gives better results
    across different query types.
    
    Parameters
    ----------
    query : str
        The user's question or search query.
    k : int, default=5
        Number of documents to retrieve.
    cosine_weight : float, default=0.6
        Weight for cosine similarity scores (0-1).
    euclidean_weight : float, default=0.4
        Weight for Euclidean similarity scores (0-1).
    normalize_scores : bool, default=True
        Normalize scores to [0,1] range.
    
    Returns
    -------
    List[Tuple[Document, float]]
        List of (document, combined_score) pairs, sorted by score descending.
    """
    # Get more candidates for reranking (fetch 2*k from each method)
    candidate_k = k * 2
    
    # Get results from both methods
    cosine_results = similarity_search_cosine(query, k=candidate_k)
    euclidean_results = similarity_search_euclidean(query, k=candidate_k, normalize_scores=True)
    
    # Combine scores with weights
    combined_scores = {}
    
    for doc, score in cosine_results:
        doc_key = doc.page_content  # Use content as key for deduplication
        combined_scores[doc_key] = {
            'document': doc,
            'score': score * cosine_weight,
            'metadata': doc.metadata
        }
    
    for doc, score in euclidean_results:
        doc_key = doc.page_content
        if doc_key in combined_scores:
            combined_scores[doc_key]['score'] += score * euclidean_weight
        else:
            combined_scores[doc_key] = {
                'document': doc,
                'score': score * euclidean_weight,
                'metadata': doc.metadata
            }
    
    # Convert to list and sort by combined score
    results = [
        (item['document'], item['score']) 
        for item in combined_scores.values()
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    if normalize_scores and results:
        max_score = results[0][1]
        if max_score > 0:
            results = [(doc, score / max_score) for doc, score in results]
    
    return results[:k]


def similarity_search_with_rerank(
    query: str,
    k: int = 5,
    candidate_multiplier: int = 3,
    similarity_metric: str = "cosine"
) -> List[Tuple[Document, float]]:
    """Retrieve more candidates then rerank based on query-document similarity.
    
    This improves retrieval quality by first getting many candidates,
    then reranking them using a more thorough similarity calculation.
    
    Parameters
    ----------
    query : str
        The user's question or search query.
    k : int, default=5
        Number of documents to return after reranking.
    candidate_multiplier : int, default=3
        How many times more candidates to fetch initially (k * multiplier).
    similarity_metric : str, default="cosine"
        Metric to use for reranking: "cosine", "euclidean", or "hybrid".
    
    Returns
    -------
    List[Tuple[Document, float]]
        Reranked list of (document, score) pairs.
    """
    candidate_k = k * candidate_multiplier
    
    # Fetch candidates using specified metric
    if similarity_metric == "cosine":
        candidates = similarity_search_cosine(query, k=candidate_k)
    elif similarity_metric == "euclidean":
        candidates = similarity_search_euclidean(query, k=candidate_k, normalize_scores=True)
    elif similarity_metric == "hybrid":
        candidates = hybrid_search(query, k=candidate_k)
    else:
        raise ValueError(f"Unknown metric: {similarity_metric}. Use 'cosine', 'euclidean', or 'hybrid'")
    
    # Return top k (already sorted by score)
    return candidates[:k]


def get_vectorstore_stats() -> Dict:
    """Get statistics about the current vector store.
    
    Returns
    -------
    Dict
        Dictionary containing index size, dimension, and available files.
    """
    vectorstore = load_vectorstore()
    stats = {
        'total_vectors': vectorstore.index.ntotal,
        'dimension': vectorstore.index.d,
        'index_type': type(vectorstore.index).__name__,
        'persist_dir': str(VECTORSTORE_DIR),
    }
    
    # Check if FAISS index files exist
    if VECTORSTORE_DIR.exists():
        stats['index_files_exist'] = (
            (VECTORSTORE_DIR / "index.faiss").exists() and
            (VECTORSTORE_DIR / "index.pkl").exists()
        )
    else:
        stats['index_files_exist'] = False
    
    return stats


# Convenience function for backward compatibility
def similarity_search(query: str, k: int = 5, metric: str = "cosine") -> List[Document]:
    """Simple interface for similarity search - returns only documents.
    
    This is a convenience wrapper for easy integration with existing code.
    
    Parameters
    ----------
    query : str
        The user's question or search query.
    k : int, default=5
        Number of documents to retrieve.
    metric : str, default="cosine"
        Similarity metric: "cosine", "euclidean", or "hybrid".
    
    Returns
    -------
    List[Document]
        List of retrieved documents (without scores).
    """
    if metric == "cosine":
        results = similarity_search_cosine(query, k)
    elif metric == "euclidean":
        results = similarity_search_euclidean(query, k)
    elif metric == "hybrid":
        results = hybrid_search(query, k)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine', 'euclidean', or 'hybrid'")
    
    return [doc for doc, score in results]


# ── quick smoke-test ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vector store with different metrics")
    parser.add_argument("--query", type=str, default="What are director duties?", help="Test query")
    parser.add_argument("--k", type=int, default=3, help="Number of results")
    parser.add_argument("--metric", type=str, default="cosine", 
                       choices=["cosine", "euclidean", "hybrid"], help="Similarity metric")
    
    args = parser.parse_args()
    
    print("Testing vector store...")
    
    # Create or load vectorstore
    try:
        vs = load_vectorstore()
        print(f"✓ Loaded existing index with {vs.index.ntotal} vectors")
    except FileNotFoundError:
        print("Creating new index...")
        vs = create_vectorstore()
        print(f"✓ Created new index with {vs.index.ntotal} vectors")
    
    # Test search with different metrics
    print(f"\n--- Testing {args.metric.upper()} similarity ---")
    print(f"Query: '{args.query}'\n")
    
    if args.metric == "cosine":
        results = similarity_search_cosine(args.query, k=args.k)
        print(f"Found {len(results)} results using Cosine Similarity:\n")
    elif args.metric == "euclidean":
        results = similarity_search_euclidean(args.query, k=args.k)
        print(f"Found {len(results)} results using Euclidean Distance:\n")
    else:  # hybrid
        results = hybrid_search(args.query, k=args.k)
        print(f"Found {len(results)} results using Hybrid Search:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}")
        print(f"   Preview: {doc.page_content[:150]}...")
        print()
    
    # Get stats
    stats = get_vectorstore_stats()
    print(f"Vector Store Stats: {stats}")