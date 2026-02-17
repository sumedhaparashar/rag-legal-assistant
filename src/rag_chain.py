"""
RAG chain with source citations — production-grade prompt.

Builds a pipeline that:
  1. Retrieves the top-k relevant chunks (with metadata) from FAISS.
  2. Formats them into a structured legal-assistant prompt that
     *requires* the LLM to cite sources in every claim.
  3. Sends the prompt to the LLM.
  4. Returns  { "answer": str, "sources": list[dict] }

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
•  The retriever is obtained from the singleton FAISS store via
   vectorstore.load_vectorstore(), so the index is loaded once per
   process and reused across all queries.
"""

from pathlib import Path
from typing import Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from config import RETRIEVER_K
from vectorstore import load_vectorstore
from llm import get_llm


# ═══════════════════════════════════════════════════════════════════
#   PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════

LEGAL_PROMPT = PromptTemplate.from_template(
    """\
### ROLE
You are an expert legal research assistant specialising in Indian law.
Your answers are used by practising lawyers, so precision and traceability
are paramount.

### INSTRUCTIONS
1. Answer the question using **ONLY** the context provided below.
   The passages may come from **different legal documents** (e.g. the
   Companies Act, SEBI reports, securities regulations, etc.).
   Synthesise information across all relevant passages to give a
   comprehensive answer.
2. For **every factual claim** in your answer, include an inline citation
   in the format:  [Source: <filename>, Page <N>]
3. If multiple context passages — even from different documents — support
   the same point, cite all of them.
4. If the context does **not** contain sufficient information to answer
   the question, respond with exactly:
   "The provided documents do not contain enough information to answer
   this question."
   Do NOT speculate or add information from outside the context.
5. Use clear, professional language.  Structure long answers with bullet
   points or numbered lists for readability.

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
    """Join retrieved chunks into a single context block.

    Each chunk is prefixed with its citation tag so the LLM can copy it
    verbatim into its answer.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        src = Path(doc.metadata.get("source", "unknown")).name
        page = int(doc.metadata.get("page", 0)) + 1        # 1-based
        parts.append(
            f"--- Passage {i} [Source: {src}, Page {page}] ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[Dict]:
    """Build a de-duplicated citation list from Document metadata."""
    sources: List[Dict] = []
    seen: set = set()
    for doc in docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        page = int(doc.metadata.get("page", 0)) + 1        # 1-based
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": src,
                "page": page,
                "snippet": doc.page_content[:200].strip(),
            })
    return sources


# ═══════════════════════════════════════════════════════════════════
#   PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def ask(question: str) -> Dict:
    """Run the full RAG pipeline and return answer + sources.

    Returns
    -------
    dict
        {
          "answer":  str,
          "sources": [{"file": str, "page": int, "snippet": str}, ...]
        }

    If no FAISS index exists yet, raises FileNotFoundError with a
    helpful message.
    """
    # 1 ── Retrieve relevant chunks
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    relevant_docs: List[Document] = retriever.invoke(question)

    # 2 ── Edge case: retriever returns nothing
    if not relevant_docs:
        return {
            "answer": (
                "The provided documents do not contain enough information "
                "to answer this question."
            ),
            "sources": [],
        }

    # 3 ── Build the filled prompt
    context_str = _format_context(relevant_docs)
    prompt_value = LEGAL_PROMPT.format(
        context=context_str,
        question=question,
    )

    # 4 ── Generate answer via LLM
    llm = get_llm()
    answer: str = llm.invoke(prompt_value)

    # 5 ── Package response
    return {
        "answer": answer.strip(),
        "sources": _extract_sources(relevant_docs),
    }


# ═══════════════════════════════════════════════════════════════════
#   CLI SMOKE-TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    q = "What powers do company directors have?"
    print(f"Question: {q}\n")
    result = ask(q)
    print(json.dumps(result, indent=2, ensure_ascii=False))
