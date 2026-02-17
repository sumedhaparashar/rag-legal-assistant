"""
LLM factory — provider-agnostic.

Supports four providers via a single ``get_llm()`` entry point:

  ┌────────────┬──────────────────────────────────────────────────┐
  │ Provider   │ When to use                                     │
  ├────────────┼──────────────────────────────────────────────────┤
  │ ollama     │ Local dev — Mistral running on your machine      │
  │ openai     │ Production — GPT-3.5 / GPT-4 via OpenAI API     │
  │ groq       │ Production — fast Mixtral/Llama via Groq API     │
  │ together   │ Production — open models via Together API        │
  └────────────┴──────────────────────────────────────────────────┘

Switching is ONE env-var change:
    LLM_PROVIDER=groq  LLM_API_KEY=gsk_...  LLM_MODEL=mixtral-8x7b-32768

No code changes in rag_chain.py or anywhere else.

Design decisions
────────────────
•  Groq, Together, and OpenAI all expose an OpenAI-compatible REST API,
   so we reuse ``ChatOpenAI`` from ``langchain_community`` (available in
   0.2.x) with a custom ``openai_api_base``.  This avoids adding any
   new dependencies.
•  We return a BaseLLM / BaseChatModel — both support ``.invoke(str)``,
   which is all ``rag_chain.py`` needs.
"""

from config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_API_KEY,
    LLM_API_BASE,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

# Provider → default API base URL (if the user didn't set LLM_API_BASE)
_DEFAULT_BASES = {
    "openai":   "https://api.openai.com/v1",
    "groq":     "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}


def get_llm():
    """Return a configured LLM instance based on ``LLM_PROVIDER``.

    The returned object supports ``.invoke(prompt_string)`` so
    ``rag_chain.py`` works identically regardless of provider.
    """
    provider = LLM_PROVIDER.lower().strip()

    # ── Local: Ollama ────────────────────────────────────────────
    if provider == "ollama":
        from langchain_community.llms import Ollama

        return Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    # ── API-based: OpenAI / Groq / Together ──────────────────────
    if provider in _DEFAULT_BASES:
        from langchain_community.chat_models import ChatOpenAI

        api_base = LLM_API_BASE or _DEFAULT_BASES[provider]

        if not LLM_API_KEY:
            raise ValueError(
                f"LLM_PROVIDER is '{provider}' but LLM_API_KEY is not set.  "
                f"Set it in your .env or environment variables."
            )

        return ChatOpenAI(
            openai_api_key=LLM_API_KEY,
            openai_api_base=api_base,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{provider}'.  "
        f"Supported: ollama, openai, groq, together"
    )
