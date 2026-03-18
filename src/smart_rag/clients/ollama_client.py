import logging
from typing import Any

import requests

from ..config import Settings
from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def generate(self, context: str, question: str) -> str | None:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": "system", "content": self._settings.system_prompt},
                {
                    "role": "user",
                    "content": f"""Контекст:
{context}

Вопрос: {question}

Ответ:""",
                },
            ],
            "stream": False,
            "keep_alive": "24h",
            "options": {"temperature": 0.3},
        }

        try:
            response = requests.post(
                self._settings.llm_api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self._settings.request_timeout_seconds,
            )

            if response.status_code == 200:
                return response.json()["message"]["content"]

            logger.error("LLM API error: %s - %s", response.status_code, response.text)
            return None

        except requests.exceptions.Timeout:
            return "Извините, сервис временно недоступен. Попробуйте позже."

        except Exception:
            logger.exception("Unexpected LLM error")
            return None
