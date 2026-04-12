from __future__ import annotations

from ..config import Settings
from .base_llm_client import BaseLLMClient


def create_llm_client(settings: Settings) -> BaseLLMClient:
    provider = settings.llm_provider.lower()

    if provider == "mistral":
        if not settings.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY is required when LLM_PROVIDER=mistral")
        from .mistral_client import MistralClient

        return MistralClient(settings)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from .openai_client import OpenAIClient

        return OpenAIClient(settings)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}. Supported providers: mistral, openai")
