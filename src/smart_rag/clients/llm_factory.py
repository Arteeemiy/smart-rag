from ..config import Settings
from .base_llm_client import BaseLLMClient
from .ollama_client import OllamaClient
from .mistral_client import MistralClient


def create_llm_client(settings: Settings) -> BaseLLMClient:

    if settings.llm_provider == "ollama":
        return OllamaClient(settings)

    if settings.llm_provider == "mistral":
        return MistralClient(settings)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
