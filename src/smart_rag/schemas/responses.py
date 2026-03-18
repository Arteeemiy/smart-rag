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


class RAGRetrieveData(BaseModel):
    has_context: bool
    sources: list[dict[str, Any]]
    retrieved: str | None = None
    context: str | None = None


class APIResponse(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
