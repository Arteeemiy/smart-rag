from functools import lru_cache

from ..clients.llm_factory import create_llm_client
from ..clients.vector_store_client import VectorStoreClient
from ..config import get_settings
from ..services.agentic_rag_service import AgenticRAGService
from ..services.document_ingestion_service import DocumentIngestionService
from ..services.exporters import ResultExportService
from ..services.llm_service import LLMService
from ..services.rag_service import RAGService
from ..services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_vector_store_client() -> VectorStoreClient:
    return VectorStoreClient(settings=get_settings())


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
    settings = get_settings()
    return DocumentIngestionService(vector_store_client=get_vector_store_client(), settings=settings)


@lru_cache(maxsize=1)
def get_result_export_service() -> ResultExportService:
    return ResultExportService(settings=get_settings())


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    settings = get_settings()
    vector_store_client = get_vector_store_client()
    retrieval_service = RetrievalService(vector_store_client=vector_store_client, settings=settings)
    llm_client = create_llm_client(settings)
    llm_service = LLMService(llm_client=llm_client)
    agentic_rag_service = AgenticRAGService(retrieval_service=retrieval_service, settings=settings)
    return RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        settings=settings,
        agentic_rag_service=agentic_rag_service,
    )
