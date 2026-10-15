"""
RAG chain with source citations — production-grade prompt with multiple retrieval strategies.

Builds a pipeline that:
  1. Retrieves the top-k relevant chunks (with metadata) from FAISS using
     configurable similarity metrics (cosine, euclidean, or hybrid).
  2. Formats them into a structured legal-assistant prompt that
     *requires* the LLM to cite sources in every claim.
  3. Sends the prompt to the LLM.
  4. Returns  { "answer": str, "sources": list[dict], "metadata": dict }
"""

import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional

# Allow running this file directly (`python src/rag_chain.py`) as well as
# as a module (`python -m src.rag_chain`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.config import RETRIEVER_K
from src.vectorstore import (
    similarity_search_cosine,
    similarity_search_euclidean,
    hybrid_search,
)
from src.llm import get_llm


# ═══════════════════════════════════════════════════════════════════
#   ENHANCED PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════

LEGAL_PROMPT = PromptTemplate.from_template(
    """\
### ROLE
You are an expert legal research assistant specialising in corporate law.
Your answers are used by practising lawyers and legal researchers, so precision,
traceability, and accuracy are paramount.

### INSTRUCTIONS
1. Answer the question using **ONLY** the context provided below.
   The passages may come from **different legal documents** (e.g., Companies Act,
   SEBI regulations, Banking Regulation Act, etc.).
   Synthesise information across all relevant passages to give a comprehensive answer.

2. For **every factual claim** in your answer, include an inline citation
   in the format:  [Source: <act name>, Page <N>]
   If a section number is available, include it:
   [Source: <act name>, Section <X>, Page <N>]

3. If multiple context passages support the same point, cite all of them.

4. If the context does **not** contain sufficient information to answer
   the question, respond with exactly:
   "Based on the provided documents, I cannot find sufficient information to answer this question."

   Do NOT speculate or add information from outside the context.

5. Structure your answer clearly:
   - Use bullet points or numbered lists for multiple points
   - Group related information under subheadings when appropriate
   - Highlight critical compliance requirements

6. End your answer with a confidence assessment based on source quality:
   - "High confidence: Direct references found in primary legislation"
   - "Medium confidence: Information synthesized across multiple sources"
   - "Low confidence: Limited or indirect references available"

### CONTEXT (retrieved from legal documents)
{context}

### QUESTION
{question}

### ANSWER
"""
)


# ═══════════════════════════════════════════════════════════════════
#   INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _format_context(docs: List[Document]) -> str:
    """Join retrieved chunks into a context block with rich legal citations."""
    parts = []
    for i, doc in enumerate(docs, 1):
        act = doc.metadata.get("act_name") or Path(
            doc.metadata.get("source", "unknown")
        ).name
        page = int(doc.metadata.get("page", 0)) + 1
        section = doc.metadata.get("section")

        if section:
            citation = f"[Source: {act}, Section {section}, Page {page}]"
        else:
            citation = f"[Source: {act}, Page {page}]"

        parts.append(
            f"--- Passage {i} {citation} ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _extract_sources(docs: List[Document], max_snippet_length: int = 300) -> List[Dict]:
    """Build a de-duplicated citation list using legal metadata.

    Each entry contains:
      - file           : act name (display)
      - source_file    : actual PDF filename on disk (for /docs/ URL)
      - page           : 1-based page number
      - snippet        : short text preview
      - section        : detected section number, if any
      - relevance_score: retrieval score, if available
    """
    sources: List[Dict] = []
    seen: set = set()

    for doc in docs:
        act = doc.metadata.get("act_name") or Path(
            doc.metadata.get("source", "unknown")
        ).name
        page = int(doc.metadata.get("page", 0)) + 1
        section = doc.metadata.get("section")
        key = (act, page, section)

        if key in seen:
            continue
        seen.add(key)

        snippet = doc.page_content.strip()
        if len(snippet) > max_snippet_length:
            snippet = snippet[:max_snippet_length] + "..."

        # Resolve the on-disk filename for building /docs/ URLs.
        source_file = doc.metadata.get("source_file")
        if not source_file:
            raw_source = doc.metadata.get("source", "")
            source_file = Path(raw_source).name if raw_source else act

        entry = {
            "file": act,
            "source_file": source_file,
            "page": page,
            "snippet": snippet,
            "relevance_score": doc.metadata.get("relevance_score"),
        }
        if section:
            entry["section"] = section
        sources.append(entry)

    return sources


def _postprocess_answer(answer: str, docs: List[Document]) -> str:
    """Clean and validate the LLM's answer."""
    if docs and not re.search(r'\[Source:.*?\]', answer):
        if "cannot find sufficient information" not in answer.lower():
            answer += "\n\n*Note: Specific page references were not included in the response.*"

    answer = re.sub(r'\](\S)', r'] \1', answer)
    return answer.strip()


# ═══════════════════════════════════════════════════════════════════
#   MAIN RAG FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def ask(
    question: str,
    k: Optional[int] = None,
    similarity_metric: str = "cosine",
    return_retrieval_metadata: bool = False,
) -> Dict:
    """Run the full RAG pipeline with configurable retrieval strategy."""
    start_total = time.time()
    k = k or RETRIEVER_K

    # 1 ── Retrieve
    retrieval_start = time.time()

    try:
        if similarity_metric == "cosine":
            docs_with_scores = similarity_search_cosine(question, k=k)
        elif similarity_metric == "euclidean":
            docs_with_scores = similarity_search_euclidean(question, k=k)
        elif similarity_metric == "hybrid":
            docs_with_scores = hybrid_search(question, k=k)
        elif similarity_metric == "auto":
            if len(question.split()) < 5:
                docs_with_scores = similarity_search_cosine(question, k=k)
            elif "section" in question.lower() or "article" in question.lower():
                docs_with_scores = hybrid_search(
                    question, k=k, cosine_weight=0.4, euclidean_weight=0.6
                )
            else:
                docs_with_scores = similarity_search_cosine(question, k=k)
        else:
            raise ValueError(
                f"Unknown similarity_metric: {similarity_metric}. "
                f"Use 'cosine', 'euclidean', 'hybrid', or 'auto'"
            )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nPlease run 'python scripts/ingest.py' first to build the FAISS index."
        )

    relevant_docs = [doc for doc, _ in docs_with_scores]
    relevance_scores = [score for _, score in docs_with_scores]
    retrieval_time = time.time() - retrieval_start

    for doc, score in zip(relevant_docs, relevance_scores):
        doc.metadata["relevance_score"] = score

    # 2 ── Edge case
    if not relevant_docs:
        return {
            "answer": (
                "Based on the provided documents, I cannot find sufficient "
                "information to answer this question. Please try rephrasing."
            ),
            "sources": [],
            "metadata": (
                {
                    "retrieval_time_seconds": round(retrieval_time, 3),
                    "num_docs_retrieved": 0,
                    "similarity_metric": similarity_metric,
                    "k_value": k,
                }
                if return_retrieval_metadata
                else None
            ),
        }

    # 3 ── Build prompt
    context_str = _format_context(relevant_docs)
    prompt_value = LEGAL_PROMPT.format(context=context_str, question=question)

    # 4 ── Generate
    generation_start = time.time()
    llm = get_llm()
    raw = llm.invoke(prompt_value)

    if hasattr(raw, "content"):
        answer = raw.content
    elif isinstance(raw, str):
        answer = raw
    else:
        answer = str(raw)

    generation_time = time.time() - generation_start

    # 5 ── Post-process
    answer = _postprocess_answer(answer, relevant_docs)
    total_time = time.time() - start_total

    # 6 ── Package
    response = {
        "answer": answer,
        "sources": _extract_sources(relevant_docs),
    }

    if return_retrieval_metadata:
        avg_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        response["metadata"] = {
            "retrieval_time_seconds": round(retrieval_time, 3),
            "generation_time_seconds": round(generation_time, 3),
            "total_time_seconds": round(total_time, 3),
            "num_docs_retrieved": len(relevant_docs),
            "similarity_metric": similarity_metric,
            "k_value": k,
            "avg_relevance_score": round(avg_score, 4),
            "min_relevance_score": round(min(relevance_scores), 4) if relevance_scores else None,
            "max_relevance_score": round(max(relevance_scores), 4) if relevance_scores else None,
        }

    return response


def ask_batch(
    questions: List[str],
    k: Optional[int] = None,
    similarity_metric: str = "cosine",
) -> List[Dict]:
    """Process multiple questions in batch."""
    results = []
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}...")
        try:
            result = ask(
                question, k=k, similarity_metric=similarity_metric,
                return_retrieval_metadata=True,
            )
            results.append(result)
        except Exception as e:
            results.append({
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)},
            })
    return results


def quick_ask(question: str) -> Dict:
    """Simplified version for API endpoints."""
    result = ask(question, similarity_metric="cosine", return_retrieval_metadata=False)
    return {"answer": result["answer"], "sources": result["sources"]}


# ═══════════════════════════════════════════════════════════════════
#   CLI SMOKE-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test RAG chain")
    parser.add_argument("--question", type=str, default="What powers do company directors have?")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "hybrid", "auto"])
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Metric  : {args.metric}")
    print(f"K       : {args.k}")
    print("-" * 80)

    try:
        result = ask(
            args.question,
            k=args.k,
            similarity_metric=args.metric,
            return_retrieval_metadata=args.verbose,
        )

        print("\nANSWER:\n")
        print(result["answer"])

        print("\nSOURCES:\n")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['file']} - Page {source['page']}")
            print(f"   file on disk: {source.get('source_file')}")
            print(f"   {source['snippet'][:100]}...")

        if args.verbose and "metadata" in result:
            print("\nMETRICS:")
            for k_, v in result["metadata"].items():
                print(f"   {k_}: {v}")

        print("\n[OK] Test completed successfully")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease run: python scripts/ingest.py")
    except Exception as e:
        print(f"\n[ERROR] {e}")