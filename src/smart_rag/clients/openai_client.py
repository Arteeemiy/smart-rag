from __future__ import annotations

import logging
import random
import time

from ..config import Settings
from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    def __init__(self, settings: Settings, model_name: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("openai package is required for OpenAIClient") from error

        self._settings = settings
        self._model_name = model_name or settings.llm_model
        self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_url)
        self._max_retries = settings.llm_max_retries
        self._base_retry_delay = settings.llm_retry_delay_seconds
        self._max_retry_delay = settings.llm_max_retry_delay_seconds
        self._request_timeout = settings.llm_timeout_seconds


    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def generate(self, context: str, question: str) -> str | None:
        return self.generate_text(
            self._settings.system_prompt,
            f"Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:",
        )

    def generate_text(self, system_prompt: str, user_prompt: str, *, model: str | None = None, temperature: float = 0.3) -> str | None:
        last_error: Exception | None = None
        model_name = model or self._model_name
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    timeout=self._request_timeout,
                )
                return response.choices[0].message.content
            except Exception as error:
                last_error = error
                error_text = str(error)
                is_retryable = any(marker in error_text.lower() for marker in ["429", "rate limit", "timeout", "timed out", "502", "503", "504"])
                logger.warning("OpenAI generation failed | model=%s | attempt=%s/%s | retryable=%s | error=%s", model_name, attempt, self._max_retries, is_retryable, error_text)
                if not is_retryable or attempt == self._max_retries:
                    break
                sleep_seconds = min(self._base_retry_delay * (2 ** (attempt - 1)), self._max_retry_delay)
                sleep_seconds += random.uniform(0, 0.3)
                time.sleep(sleep_seconds)
        logger.exception("OpenAI generation failed after retries", exc_info=last_error)
        return None
