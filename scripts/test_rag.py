"""
Full RAG pipeline tester — retrieval + LLM + citations.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.rag_chain import ask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="What are the duties of company directors?")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "hybrid"])
    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Metric  : {args.metric}")
    print(f"K       : {args.k}")
    print("-" * 70)

    result = ask(
        args.question,
        k=args.k,
        similarity_metric=args.metric,
        return_retrieval_metadata=True,
    )

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:\n")
    for i, s in enumerate(result["sources"], 1):
        section = s.get("section", "-")
        print(f"{i}. {s['file']} | page {s['page']} | section {section}")
        print(f"   {s['snippet'][:120]}...")

    if "metadata" in result:
        print("\nMETRICS:")
        for k, v in result["metadata"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()