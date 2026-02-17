"""
Embedding model factory.

Returns a *reusable* HuggingFaceEmbeddings instance.
This module no longer loads documents — that was the old coupling.

Design decision:  We use a module-level singleton (_model) so the
heavy model download / load happens at most once per process,
regardless of how many callers import this.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME

_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a singleton HuggingFaceEmbeddings instance."""
    global _model
    if _model is None:
        _model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
        )
    return _model
