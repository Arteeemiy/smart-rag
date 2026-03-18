from functools import lru_cache

from ..clients.llm_factory import create_llm_client
from ..clients.vector_store_client import VectorStoreClient
from ..config import get_settings
from ..services.llm_service import LLMService
from ..services.rag_service import RAGService
from ..services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    settings = get_settings()

    vector_store_client = VectorStoreClient(settings=settings)

    retrieval_service = RetrievalService(
        vector_store_client=vector_store_client,
        settings=settings,
    )

    llm_client = create_llm_client(settings)

    llm_service = LLMService(llm_client=llm_client)

    return RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service,
    )
