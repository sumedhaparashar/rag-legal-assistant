"""
Quick RAG retrieval tester.

Usage:
    python scripts\test_retrieval.py
    python scripts\test_retrieval.py --query "What are director duties?" --k 3
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.vectorstore import (
    load_vectorstore,
    similarity_search_cosine,
    similarity_search_euclidean,
    hybrid_search,
    get_vectorstore_stats,
)


def run(question: str, k: int, metric: str):
    print("Loading vector store...")
    vs = load_vectorstore()
    stats = get_vectorstore_stats()
    print(f"Vectors indexed : {stats['total_vectors']}")
    print(f"Dimension       : {stats['dimension']}")
    print(f"Index type      : {stats['index_type']}")
    print()

    if metric == "cosine":
        results = similarity_search_cosine(question, k=k)
    elif metric == "euclidean":
        results = similarity_search_euclidean(question, k=k)
    elif metric == "hybrid":
        results = hybrid_search(question, k=k)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    print(f"--- Query: {question!r}")
    print(f"--- Metric: {metric}  |  k: {k}")
    print()

    for i, (doc, score) in enumerate(results, 1):
        act = doc.metadata.get("act_name", doc.metadata.get("source", "?"))
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "-")
        print(f"{i}. score={score:.4f}")
        print(f"   source  : {act}")
        print(f"   page    : {page}")
        print(f"   section : {section}")
        print(f"   preview : {doc.page_content[:160].strip()}...")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What are the duties of directors?")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean", "hybrid"],
    )
    args = parser.parse_args()
    run(args.query, args.k, args.metric)


if __name__ == "__main__":
    main()