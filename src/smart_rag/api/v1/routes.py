import http
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ...core.dependencies import get_rag_service
from ...schemas.requests import RAGAskRequest, RAGRetrieveRequest
from ...services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


def create_json_response(data: dict[str, Any] | None, status: int = 200, error: str | None = None) -> JSONResponse:
    return JSONResponse(
        content={
            "success": http.HTTPStatus.OK.value <= status < http.HTTPStatus.MULTIPLE_CHOICES.value,
            "data": data,
            "error": error,
        },
        status_code=status,
    )


@router.get("/ping", summary="Health check endpoint")
async def get_ping(request: Request) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("Ping request received from %s", client_host)
    return create_json_response(data={"data": "pong"})


@router.post("/rag/ask")
async def rag_ask(
    payload: RAGAskRequest,
    request: Request,
    rag_service: RAGService = Depends(get_rag_service),
) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("RAG query from %s", client_host)
    try:
        result = rag_service.ask(payload.query, top_k=payload.top_k)
        if not payload.debug:
            result.pop("retrieved", None)
        return create_json_response(data=result)
    except Exception as error:
        logger.exception("RAG ask failed")
        return create_json_response(data=None, status=500, error=str(error))


@router.post("/rag/retrieve")
async def rag_retrieve(
    payload: RAGRetrieveRequest,
    request: Request,
    rag_service: RAGService = Depends(get_rag_service),
) -> JSONResponse:
    client_host = request.client.host if request.client else "unknown"
    logger.info("RAG retrieve from %s", client_host)
    try:
        context, sources, retrieved = rag_service.retrieve(payload.query, top_k=payload.top_k)
        result = {
            "has_context": bool(context) and bool(sources),
            "sources": sources,
            "context": context,
            "retrieved": retrieved if payload.debug else None,
        }
        return create_json_response(data=result)
    except Exception as error:
        logger.exception("RAG retrieve failed")
        return create_json_response(data=None, status=500, error=str(error))
