from ..clients.base_llm_client import BaseLLMClient


class LLMService:
    def __init__(self, llm_client: BaseLLMClient) -> None:
        self._llm_client = llm_client

    def generate_response(self, context: str, question: str) -> str | None:
        return self._llm_client.generate(context=context, question=question)
