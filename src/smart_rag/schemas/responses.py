from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SourceItem(BaseModel):
    section: str | None = None
    source: str | None = None
    distance: float | None = None


class RAGAskData(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    has_context: bool
    error: str | None = None
    retrieved: str | None = None
    execution_mode: str | None = None


class RAGRetrieveData(BaseModel):
    has_context: bool
    sources: list[dict[str, Any]]
    retrieved: str | None = None
    context: str | None = None
    execution_mode: str | None = None


class RetrievalDiagnostics(BaseModel):
    accepted: bool
    reason: str
    query_word_count: int
    requested_top_k: int
    top_hit_distance: float | None = None
    second_hit_distance: float | None = None
    gap12: float | None = None
    cutoff_distance: float | None = None
    total_hits: int
    selected_hits: int
    thresholds: dict[str, Any]


class TestQuestionResult(BaseModel):
    question_id: str | None = None
    query: str
    answer: str | None = None
    has_context: bool
    error: str | None = None
    sources: list[dict[str, Any]]
    context: str | None = None
    retrieved: str | None = None
    retrieval_diagnostics: RetrievalDiagnostics
    hit_details: list[dict[str, Any]]
    metadata: dict[str, Any] | None = None
    execution_mode: str | None = None


class TestRunSummary(BaseModel):
    total_questions: int
    answered_questions: int
    questions_with_context: int
    questions_without_context: int
    questions_with_error: int


class APIResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
