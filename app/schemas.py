"""
Pydantic request / response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List


# ── /ask ─────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The legal question to answer.",
        json_schema_extra={"example": "What powers do company directors have?"},
    )


class SourceInfo(BaseModel):
    file: str = Field(..., description="Source PDF filename")
    page: int = Field(..., description="1-based page number")
    snippet: str = Field(..., description="First ~200 chars of the chunk")


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []


# ── /ingest ──────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str = "success"
    chunks_indexed: int = 0


# ── /auto-ingest ─────────────────────────────────────────────────

class AutoIngestResponse(BaseModel):
    status: str = "success"
    new_chunks: int = 0
    reused_chunks: int = 0
    pruned_chunks: int = 0
    total_vectors: int = 0
    elapsed_seconds: float = 0.0

