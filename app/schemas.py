"""
Pydantic request / response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ── /ask ─────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The legal question to answer.",
        json_schema_extra={"example": "What powers do company directors have?"},
    )


class SourceInfo(BaseModel):
    file: str = Field(..., description="Display name of the source (act name)")
    source_file: Optional[str] = Field(
        None,
        description="Actual PDF filename on disk (used to build /docs/ URL)",
    )
    page: int = Field(..., description="1-based page number")
    snippet: str = Field(..., description="First ~300 chars of the chunk")
    section: Optional[str] = Field(
        None, description="Detected section/article number, if any"
    )
    relevance_score: Optional[float] = Field(
        None, description="Similarity score (0-1)"
    )


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


# ── /api/auth/config ─────────────────────────────────────────────

class AuthConfigResponse(BaseModel):
    supabase_url: str = ""
    supabase_anon_key: str = ""


# ── /api/analytics ───────────────────────────────────────────────

class TopicCount(BaseModel):
    topic: str
    count: int


class DailyCount(BaseModel):
    date: str
    count: int


class AnalyticsResponse(BaseModel):
    total_users: int = 0
    daily_active_users: int = 0
    total_questions: int = 0
    weekly_questions: int = 0
    avg_response_time_ms: float = 0.0
    top_topics: List[TopicCount] = []
    uptime_seconds: int = 0
    daily_question_counts: List[DailyCount] = []