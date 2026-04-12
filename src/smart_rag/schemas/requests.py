from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


RagMode = str
ResponseMode = Literal["full", "summary"]
ExportFormat = Literal["none", "json", "google_sheets"]


class GoogleSheetsExportConfig(BaseModel):
    credentials_json: str | None = Field(
        default=None,
        description="Service account JSON content as a string. Optional when GOOGLE_SHEETS_CREDENTIALS_FILE is configured.",
    )
    spreadsheet_id: str | None = Field(default=None, description="Existing Google Spreadsheet ID.")
    spreadsheet_title: str | None = Field(default=None, description="New spreadsheet title to create when spreadsheet_id is not provided.")
    worksheet_name: str | None = Field(default="results", description="Worksheet name for detailed rows.")
    summary_worksheet_name: str | None = Field(default="summary", description="Worksheet name for run summary.")
    share_with_email: str | None = Field(default=None, description="Optional email to share the spreadsheet with.")


class ExportRequest(BaseModel):
    format: ExportFormat = Field(default="none")
    filename_prefix: str | None = Field(default=None, description="Filename/title prefix for exported artifacts.")
    google_sheets: GoogleSheetsExportConfig | None = None

    @model_validator(mode="after")
    def validate_export(self) -> "ExportRequest":
        if self.format == "google_sheets" and self.google_sheets is None:
            raise ValueError("google_sheets config is required when export format is google_sheets")
        return self


class RAGAskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    top_k: int | None = Field(default=5, ge=1, le=20)
    debug: bool = False
    rag_mode: RagMode | None = None
    generation_profile: str | None = None


class RAGRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=5, ge=1, le=20)
    debug: bool = True
    rag_mode: RagMode | None = None
    generation_profile: str | None = None
    retrieval_mode: str = Field(default="smart", description="Retrieval strategy: smart or vector")
    retrieval_overrides: RetrievalOverrides | None = None


class RetrievalOverrides(BaseModel):
    gate_strong: float | None = Field(default=None, ge=0)
    gate_max: float | None = Field(default=None, ge=0)
    gap_min: float | None = Field(default=None, ge=0)
    window: float | None = Field(default=None, ge=0)
    min_keep: int | None = Field(default=None, ge=1, le=20)
    max_keep: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def validate_keep_bounds(self) -> "RetrievalOverrides":
        if self.min_keep is not None and self.max_keep is not None and self.min_keep > self.max_keep:
            raise ValueError("min_keep cannot be greater than max_keep")
        return self


class QuestionTestItem(BaseModel):
    question_id: str | None = None
    query: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class RAGTestRunRequest(BaseModel):
    queries: list[str] | None = None
    items: list[QuestionTestItem] | None = None
    top_k: int | None = Field(default=5, ge=1, le=20)
    include_answers: bool = True
    include_context: bool = True
    include_retrieved_text: bool = True
    enable_query_expansion: bool = True
    enable_reranking: bool = True
    query_expansion_variants: int | None = Field(default=None, ge=1, le=10)
    rerank_top_n: int | None = Field(default=None, ge=1, le=20)
    retrieval_overrides: RetrievalOverrides | None = None
    rag_mode: RagMode | None = None
    generation_profile: str | None = None
    response_mode: ResponseMode | None = None
    inline_result_limit: int = Field(default=20, ge=1, le=200)
    export: ExportRequest | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "RAGTestRunRequest":
        has_queries = bool(self.queries)
        has_items = bool(self.items)
        if has_queries == has_items:
            raise ValueError("Provide exactly one of `queries` or `items`")
        return self
