import logging

try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral as Mistral

from ..config import Settings
from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class MistralClient(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = Mistral(api_key=settings.mistral_api_key)

    def generate(self, context: str, question: str) -> str | None:
        try:
            response = self._client.chat.complete(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": self._settings.system_prompt},
                    {
                        "role": "user",
                        "content": f"""Контекст:
{context}

Вопрос: {question}

Ответ:""",
                    },
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content

        except Exception:
            logger.exception("Mistral generation failed")
            return None
