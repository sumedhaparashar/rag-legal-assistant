"""
RAG chain with source citations — production-grade prompt with multiple retrieval strategies.

Builds a pipeline that:
  1. Retrieves the top-k relevant chunks (with metadata) from FAISS using 
     configurable similarity metrics (cosine, euclidean, or hybrid).
  2. Formats them into a structured legal-assistant prompt that
     *requires* the LLM to cite sources in every claim.
  3. Sends the prompt to the LLM.
  4. Returns  { "answer": str, "sources": list[dict], "metadata": dict }

Design decisions
────────────────
•  The prompt is deliberately prescriptive: it tells the LLM the exact
   citation format to use ("[Source: file, Page N]") so that answers are
   consistently traceable.
•  A "NOT FOUND" instruction prevents hallucination — if the context
   doesn't answer the question, the model must say so explicitly instead
   of guessing.
•  We use manual LCEL (not deprecated RetrievalQA) so we can return the
   source documents alongside the generated answer.
•  Supports multiple similarity metrics for research comparison.
•  Includes retrieval performance metrics for evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.config import RETRIEVER_K
from src.vectorstore import (
    load_vectorstore, 
    similarity_search_cosine,
    similarity_search_euclidean,
    hybrid_search,
    similarity_search
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
   in the format:  [Source: <filename>, Page <N>]
   
3. If multiple context passages support the same point, cite all of them.
   Example: "Directors must file annual returns [Source: companies_act.pdf, Page 42]
   [Source: sebi_guidelines.pdf, Page 15]"

4. If you cite a specific section or article number mentioned in the context,
   include it: "[Source: companies_act.pdf, Page 35, Section 149]"

5. If the context does **not** contain sufficient information to answer
   the question, respond with exactly:
   "Based on the provided documents, I cannot find sufficient information to answer this question."
   
   Do NOT speculate or add information from outside the context.

6. For definitions or key legal concepts, quote the exact text when possible:
   "As defined in the act: 'director includes...' [Source: companies_act.pdf, Page 28]"

7. Structure your answer clearly:
   - Use bullet points or numbered lists for multiple points
   - Group related information under subheadings when appropriate
   - Highlight critical compliance requirements

8. End your answer with a confidence assessment based on source quality:
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
    """Join retrieved chunks into a single context block with rich metadata.
    
    Each chunk is prefixed with its citation tag so the LLM can copy it
    verbatim into its answer. Includes section numbers when available.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        src = Path(doc.metadata.get("source", "unknown")).name
        page = int(doc.metadata.get("page", 0)) + 1        # 1-based page numbers
        
        # Build citation with optional section info
        citation = f"[Source: {src}, Page {page}]"
        
        # Try to extract section/heading from content for better context
        content_preview = doc.page_content
        first_line = content_preview.split('\n')[0][:100]
        if re.match(r'^(Section|Chapter|Article|PART)\s+\d+', first_line, re.I):
            citation = f"[Source: {src}, Page {page}, {first_line[:50]}]"
        
        parts.append(
            f"--- Passage {i} {citation} ---\n"
            f"{content_preview}"
        )
    return "\n\n".join(parts)


def _extract_sources(docs: List[Document], max_snippet_length: int = 300) -> List[Dict]:
    """Build a de-duplicated citation list from Document metadata.
    
    Parameters
    ----------
    docs : List[Document]
        Retrieved documents
    max_snippet_length : int
        Maximum length of snippet text (default 300 chars)
    
    Returns
    -------
    List[Dict]
        List of unique sources with file, page, and content snippets
    """
    sources: List[Dict] = []
    seen: set = set()
    
    for doc in docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        page = int(doc.metadata.get("page", 0)) + 1        # 1-based page numbers
        key = (src, page)
        
        if key not in seen:
            seen.add(key)
            
            # Clean and truncate snippet
            snippet = doc.page_content.strip()
            if len(snippet) > max_snippet_length:
                snippet = snippet[:max_snippet_length] + "..."
            
            sources.append({
                "file": src,
                "page": page,
                "snippet": snippet,
                "relevance_score": doc.metadata.get("relevance_score", None)
            })
    
    return sources


def _postprocess_answer(answer: str, docs: List[Document]) -> str:
    """Clean and validate the LLM's answer.
    
    - Ensures citations are properly formatted
    - Removes any hallucinated content markers
    - Adds source summary if missing citations
    """
    # Check if answer has any citations when it should
    if docs and not re.search(r'\[Source:.*?\]', answer):
        # Add a note about missing citations (helpful for debugging)
        if "cannot find sufficient information" not in answer.lower():
            answer += "\n\n*Note: Specific page references were not included in the response.*"
    
    # Ensure proper spacing after citations
    answer = re.sub(r'\](\S)', r'] \1', answer)
    
    return answer.strip()


# ═══════════════════════════════════════════════════════════════════
#   MAIN RAG FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def ask(
    question: str, 
    k: Optional[int] = None,
    similarity_metric: str = "cosine",
    return_retrieval_metadata: bool = False
) -> Dict:
    """Run the full RAG pipeline with configurable retrieval strategy.
    
    Parameters
    ----------
    question : str
        User's legal question
    k : int, optional
        Number of documents to retrieve (defaults to config.RETRIEVER_K)
    similarity_metric : str, default="cosine"
        Similarity metric for retrieval: "cosine", "euclidean", or "hybrid"
    return_retrieval_metadata : bool, default=False
        If True, includes retrieval scores and timing in response
    
    Returns
    -------
    dict
        {
          "answer": str,
          "sources": [{"file": str, "page": int, "snippet": str}, ...],
          "metadata": {  # only if return_retrieval_metadata=True
              "retrieval_time": float,
              "generation_time": float,
              "total_time": float,
              "num_docs_retrieved": int,
              "similarity_metric": str,
              "k_value": int,
              "avg_relevance_score": float
          }
        }
    
    Raises
    ------
    FileNotFoundError
        If no FAISS index exists yet.
    """
    start_total = time.time()
    k = k or RETRIEVER_K
    
    # 1 ── Retrieve relevant chunks using specified metric
    retrieval_start = time.time()
    
    try:
        if similarity_metric == "cosine":
            docs_with_scores = similarity_search_cosine(question, k=k)
            relevant_docs = [doc for doc, score in docs_with_scores]
            relevance_scores = [score for doc, score in docs_with_scores]
            
        elif similarity_metric == "euclidean":
            docs_with_scores = similarity_search_euclidean(question, k=k)
            relevant_docs = [doc for doc, score in docs_with_scores]
            relevance_scores = [score for doc, score in docs_with_scores]
            
        elif similarity_metric == "hybrid":
            docs_with_scores = hybrid_search(question, k=k)
            relevant_docs = [doc for doc, score in docs_with_scores]
            relevance_scores = [score for doc, score in docs_with_scores]
            
        elif similarity_metric == "auto":
            # Auto-select based on query length and type
            if len(question.split()) < 5:
                # Short queries often benefit from cosine
                docs_with_scores = similarity_search_cosine(question, k=k)
            elif "section" in question.lower() or "article" in question.lower():
                # Specific section references - use hybrid
                docs_with_scores = hybrid_search(question, k=k, cosine_weight=0.4, euclidean_weight=0.6)
            else:
                # Default to cosine for general questions
                docs_with_scores = similarity_search_cosine(question, k=k)
            
            relevant_docs = [doc for doc, score in docs_with_scores]
            relevance_scores = [score for doc, score in docs_with_scores]
            
        else:
            raise ValueError(f"Unknown similarity_metric: {similarity_metric}. "
                           f"Use 'cosine', 'euclidean', 'hybrid', or 'auto'")
    
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nPlease run 'python scripts/ingest.py' first to build the FAISS index."
        )
    
    retrieval_time = time.time() - retrieval_start
    
    # Add relevance scores to document metadata for tracking
    for doc, score in zip(relevant_docs, relevance_scores):
        doc.metadata['relevance_score'] = score
    
    # 2 ── Edge case: retriever returns nothing
    if not relevant_docs:
        return {
            "answer": (
                "Based on the provided documents, I cannot find sufficient information "
                "to answer this question. Please try rephrasing or check if the "
                "documents contain relevant information."
            ),
            "sources": [],
            "metadata": {
                "retrieval_time": retrieval_time,
                "num_docs_retrieved": 0,
                "similarity_metric": similarity_metric,
                "k_value": k
            } if return_retrieval_metadata else None
        }
    
    # 3 ── Build the filled prompt
    context_str = _format_context(relevant_docs)
    prompt_value = LEGAL_PROMPT.format(
        context=context_str,
        question=question,
    )
    
    # 4 ── Generate answer via LLM
    generation_start = time.time()
    llm = get_llm()
    raw = llm.invoke(prompt_value)
    
    # Handle different LLM response types
    if hasattr(raw, "content"):
        answer = raw.content
    elif isinstance(raw, str):
        answer = raw
    else:
        answer = str(raw)
    
    generation_time = time.time() - generation_start
    
    # 5 ── Post-process answer
    answer = _postprocess_answer(answer, relevant_docs)
    total_time = time.time() - start_total
    
    # 6 ── Package response
    response = {
        "answer": answer,
        "sources": _extract_sources(relevant_docs),
    }
    
    # Add metadata if requested
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
    similarity_metric: str = "cosine"
) -> List[Dict]:
    """Process multiple questions in batch (useful for evaluation).
    
    Parameters
    ----------
    questions : List[str]
        List of questions to answer
    k : int, optional
        Number of documents to retrieve per question
    similarity_metric : str, default="cosine"
        Similarity metric for retrieval
    
    Returns
    -------
    List[Dict]
        List of response dictionaries for each question
    """
    results = []
    for i, question in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}...")
        try:
            result = ask(question, k=k, similarity_metric=similarity_metric, 
                        return_retrieval_metadata=True)
            results.append(result)
        except Exception as e:
            results.append({
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)}
            })
    return results


# ═══════════════════════════════════════════════════════════════════
#   SIMPLIFIED API FOR FASTAPI INTEGRATION
# ═══════════════════════════════════════════════════════════════════

def quick_ask(question: str) -> Dict:
    """Simplified version for API endpoints - returns answer and sources only.
    
    This is a wrapper around ask() that provides just the essential data
    for the JSON response, perfect for your /ask endpoint.
    
    Parameters
    ----------
    question : str
        User's legal question
    
    Returns
    -------
    dict
        {"answer": str, "sources": list}
    """
    result = ask(question, similarity_metric="cosine", return_retrieval_metadata=False)
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


# ═══════════════════════════════════════════════════════════════════
#   CLI SMOKE-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG chain with different metrics")
    parser.add_argument("--question", type=str, default="What powers do company directors have?", 
                       help="Question to ask")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--metric", type=str, default="cosine",
                       choices=["cosine", "euclidean", "hybrid", "auto"],
                       help="Similarity metric to use")
    parser.add_argument("--verbose", action="store_true", help="Show retrieval metadata")
    
    args = parser.parse_args()
    
    print(f"Question: {args.question}")
    print(f"Similarity Metric: {args.metric.upper()}")
    print(f"Retrieval K: {args.k}")
    print("-" * 80)
    
    try:
        result = ask(
            args.question, 
            k=args.k, 
            similarity_metric=args.metric,
            return_retrieval_metadata=args.verbose
        )
        
        print("\n📝 ANSWER:\n")
        print(result["answer"])
        
        print("\n📚 SOURCES:\n")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['file']} - Page {source['page']}")
            print(f"   {source['snippet'][:100]}...")
        
        if args.verbose and "metadata" in result:
            print("\n📊 METRICS:")
            metadata = result["metadata"]
            print(f"   Retrieval time: {metadata['retrieval_time_seconds']}s")
            print(f"   Generation time: {metadata['generation_time_seconds']}s")
            print(f"   Total time: {metadata['total_time_seconds']}s")
            print(f"   Avg relevance score: {metadata['avg_relevance_score']}")
            print(f"   Documents retrieved: {metadata['num_docs_retrieved']}")
        
        print("\n✅ Test completed successfully")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease run: python scripts/ingest.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")