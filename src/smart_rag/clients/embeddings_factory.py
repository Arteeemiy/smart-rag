from __future__ import annotations

from typing import Any

from ..config import Settings


SupportedEmbeddings = Any


def create_embeddings(settings: Settings) -> SupportedEmbeddings:
    provider = settings.embedding_provider.lower().strip()

    if provider in {"huggingface", "hf", "sentence_transformers", "sentence-transformers"}:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "model": settings.embedding_model,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": settings.llm_max_retries,
        }
        if settings.openai_api_url:
            kwargs["base_url"] = settings.openai_api_url
        return OpenAIEmbeddings(**kwargs)

    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}. "
        "Supported providers: huggingface, openai"
    )
