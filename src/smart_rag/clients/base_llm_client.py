from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @property
    def provider_name(self) -> str:
        return self.__class__.__name__.replace("Client", "").lower()

    @property
    def model_name(self) -> str | None:
        return None

    @abstractmethod
    def generate(self, context: str, question: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> str | None:
        raise NotImplementedError
