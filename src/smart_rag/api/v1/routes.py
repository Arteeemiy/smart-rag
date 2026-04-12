from __future__ import annotations

import http
import inspect
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from ...core.dependencies import get_document_ingestion_service, get_rag_service, get_result_export_service
from ...schemas.requests import (
    ExportRequest,
    GoogleSheetsExportConfig,
    QuestionTestItem,
    RAGAskRequest,
    RAGRetrieveRequest,
    RAGTestRunRequest,
    RetrievalOverrides,
)
from ...services.exporters import ResultExportService
from ...services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


def _resolve_response_mode(request: Request, requested_mode: str | None) -> str:
    if requested_mode in {"full", "summary"}:
        return requested_mode
    user_agent = (request.headers.get("user-agent") or "").lower()
    if "mozilla/" in user_agent or "swagger" in user_agent:
        return "summary"
    return "full"


def create_json_response(data: dict[str, Any] | None, status: int = 200, error: str | None = None) -> JSONResponse:
    return JSONResponse(
        content={
            "success": http.HTTPStatus.OK.value <= status < http.HTTPStatus.MULTIPLE_CHOICES.value,
            "data": data,
            "error": error,
        },
        status_code=status,
    )


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _build_test_items_from_lines(lines: list[str]) -> list[QuestionTestItem]:
    return [QuestionTestItem(question_id=str(index), query=line) for index, line in enumerate((item.strip() for item in lines), start=1) if line]


def _build_export_request(
    export_format: str,
    filename_prefix: str | None,
    credentials_json: str | None,
    spreadsheet_id: str | None,
    spreadsheet_title: str | None,
    worksheet_name: str | None,
    summary_worksheet_name: str | None,
    share_with_email: str | None,
) -> ExportRequest | None:
    if export_format == "none":
        return None
    google_sheets = None
    if export_format == "google_sheets":
        google_sheets = GoogleSheetsExportConfig(
            credentials_json=credentials_json,
            spreadsheet_id=spreadsheet_id,
            spreadsheet_title=spreadsheet_title,
            worksheet_name=worksheet_name or "results",
            summary_worksheet_name=summary_worksheet_name or "summary",
            share_with_email=share_with_email,
        )
    return ExportRequest(format=export_format, filename_prefix=filename_prefix, google_sheets=google_sheets)


@router.get("/ping", summary="Health check endpoint")
async def get_ping(request: Request) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("Ping request received from %s", client_host)
    return create_json_response(data={"data": "pong"})


@router.post("/rag/ask")
async def rag_ask(payload: RAGAskRequest, request: Request, rag_service: RAGService = Depends(get_rag_service)) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("RAG query from %s", client_host)
    try:
        result = rag_service.ask(payload.query, top_k=payload.top_k, rag_mode=payload.rag_mode, generation_profile=payload.generation_profile)
        if not payload.debug:
            result.pop("retrieved", None)
        return create_json_response(data=result)
    except Exception as error:
        logger.exception("RAG ask failed")
        return create_json_response(data=None, status=500, error=str(error))


@router.post("/rag/retrieve")
async def rag_retrieve(payload: RAGRetrieveRequest, request: Request, rag_service: RAGService = Depends(get_rag_service)) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("RAG retrieve from %s", client_host)
    try:
        detailed = rag_service.retrieve_details(
            question=payload.query,
            top_k=payload.top_k,
            retrieval_overrides=payload.retrieval_overrides,
            include_context=True,
            include_retrieved_text=payload.debug,
            retrieval_mode=payload.retrieval_mode,
        )
        result = {
            "has_context": detailed["has_context"],
            "sources": detailed["sources"],
            "context": detailed["context"],
            "retrieved": detailed["retrieved"] if payload.debug else None,
            "retrieval_diagnostics": detailed["retrieval_diagnostics"],
            "hit_details": detailed["hit_details"] if payload.debug else None,
            "rag_mode": detailed.get("rag_mode"),
        }
        return create_json_response(data=result)
    except Exception as error:
        logger.exception("RAG retrieve failed")
        return create_json_response(data=None, status=500, error=str(error))


@router.post("/rag/test/run")
async def rag_test_run(
    payload: RAGTestRunRequest,
    request: Request,
    rag_service: RAGService = Depends(get_rag_service),
    result_export_service: ResultExportService = Depends(get_result_export_service),
) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("RAG test run from %s", client_host)
    try:
        items = payload.items or _build_test_items_from_lines(payload.queries or [])
        run_kwargs = _filter_supported_kwargs(
            rag_service.run_test_queries,
            {
                "items": items,
                "top_k": payload.top_k,
                "retrieval_overrides": payload.retrieval_overrides,
                "include_answers": payload.include_answers,
                "include_context": payload.include_context,
                "include_retrieved_text": payload.include_retrieved_text,
                "enable_query_expansion": payload.enable_query_expansion,
                "enable_reranking": payload.enable_reranking,
                "query_expansion_variants": payload.query_expansion_variants,
                "rerank_top_n": payload.rerank_top_n,
                "rag_mode": payload.rag_mode,
                "generation_profile": payload.generation_profile,
            },
        )
        run_result = rag_service.run_test_queries(**run_kwargs)
        export_info = result_export_service.export_run_result(run_result, payload.export)
        response_payload = result_export_service.build_response_payload(
            run_result=run_result,
            response_mode=_resolve_response_mode(request, payload.response_mode),
            inline_result_limit=payload.inline_result_limit,
            export_info=export_info,
        )
        return create_json_response(data=response_payload)
    except Exception as error:
        logger.exception("RAG test run failed")
        return create_json_response(data=None, status=500, error=str(error))


@router.post("/rag/test/run-file")
async def rag_test_run_file(
    request: Request,
    file: UploadFile = File(...),
    document: UploadFile | None = File(default=None),
    top_k: int = Form(default=5),
    include_answers: bool = Form(default=True),
    include_context: bool = Form(default=True),
    include_retrieved_text: bool = Form(default=True),
    gate_strong: float | None = Form(default=None),
    gate_max: float | None = Form(default=None),
    gap_min: float | None = Form(default=None),
    window: float | None = Form(default=None),
    min_keep: int | None = Form(default=None),
    max_keep: int | None = Form(default=None),
    enable_query_expansion: bool = Form(default=True),
    enable_reranking: bool = Form(default=True),
    query_expansion_variants: int | None = Form(default=None),
    rerank_top_n: int | None = Form(default=None),
    rag_mode: str | None = Form(default=None),
    generation_profile: str | None = Form(default=None),
    response_mode: str | None = Form(default=None),
    inline_result_limit: int = Form(default=20),
    export_format: str = Form(default="none"),
    export_filename_prefix: str | None = Form(default=None),
    google_credentials_json: str | None = Form(default=None),
    google_spreadsheet_id: str | None = Form(default=None),
    google_spreadsheet_title: str | None = Form(default=None),
    google_worksheet_name: str | None = Form(default="results"),
    google_summary_worksheet_name: str | None = Form(default="summary"),
    google_share_with_email: str | None = Form(default=None),
    rag_service: RAGService = Depends(get_rag_service),
    result_export_service: ResultExportService = Depends(get_result_export_service),
) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("RAG test file run from %s", client_host)
    try:
        content = (await file.read()).decode("utf-8")
        items = _build_test_items_from_lines(content.splitlines())
        overrides = RetrievalOverrides(
            gate_strong=gate_strong,
            gate_max=gate_max,
            gap_min=gap_min,
            window=window,
            min_keep=min_keep,
            max_keep=max_keep,
        )
        uploaded_document = None
        vector_store_override = None
        retrieval_scope = "knowledge_base"
        if document is not None:
            document_ingestion_service = get_document_ingestion_service()
            uploaded_document = document_ingestion_service.build_uploaded_knowledge_base(
                filename=document.filename or "uploaded_document",
                file_bytes=await document.read(),
                content_type=document.content_type,
            )
            vector_store_override = uploaded_document.vector_store
            retrieval_scope = "uploaded_document"
        run_kwargs = _filter_supported_kwargs(
            rag_service.run_test_queries,
            {
                "items": items,
                "top_k": top_k,
                "retrieval_overrides": overrides,
                "include_answers": include_answers,
                "include_context": include_context,
                "include_retrieved_text": include_retrieved_text,
                "enable_query_expansion": enable_query_expansion,
                "enable_reranking": enable_reranking,
                "query_expansion_variants": query_expansion_variants,
                "rerank_top_n": rerank_top_n,
                "rag_mode": rag_mode,
                "vector_store_override": vector_store_override,
                "retrieval_scope": retrieval_scope,
                "uploaded_document_info": (
                    {
                        "filename": uploaded_document.filename,
                        "content_type": uploaded_document.content_type,
                        "chunk_count": uploaded_document.chunk_count,
                    }
                    if uploaded_document is not None
                    else None
                ),
            },
        )
        run_result = rag_service.run_test_queries(**run_kwargs)
        run_result["input_file"] = {"filename": file.filename, "content_type": file.content_type}
        export_request = _build_export_request(
            export_format=export_format,
            filename_prefix=export_filename_prefix,
            credentials_json=google_credentials_json,
            spreadsheet_id=google_spreadsheet_id,
            spreadsheet_title=google_spreadsheet_title,
            worksheet_name=google_worksheet_name,
            summary_worksheet_name=google_summary_worksheet_name,
            share_with_email=google_share_with_email,
        )
        export_info = result_export_service.export_run_result(run_result, export_request)
        response_payload = result_export_service.build_response_payload(
            run_result=run_result,
            response_mode=_resolve_response_mode(request, response_mode),
            inline_result_limit=inline_result_limit,
            export_info=export_info,
        )
        if response_mode != "full":
            response_payload["input_file"] = run_result["input_file"]
        return create_json_response(data=response_payload)
    except Exception as error:
        logger.exception("RAG test file run failed")
        return create_json_response(data=None, status=500, error=str(error))
