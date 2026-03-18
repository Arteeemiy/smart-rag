from typing import Any

from .llm_service import LLMService
from .retrieval_service import RetrievalService


class RAGService:
    def __init__(self, retrieval_service: RetrievalService, llm_service: LLMService) -> None:
        self._retrieval_service = retrieval_service
        self._llm_service = llm_service

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        if not question.strip():
            return {
                "answer": "Пожалуйста, задайте вопрос.",
                "sources": [],
                "has_context": False,
                "error": None,
                "retrieved": "",
            }

        context, sources, retrieved = self._retrieval_service.retrieve_context(question, top_k=top_k)
        if not context or "Нет релевантной информации" in context:
            return {
                "answer": (
                    "К сожалению, в моей базе знаний нет информации по этому вопросу. "
                    "Обратитесь к технической поддержке."
                ),
                "sources": [],
                "has_context": False,
                "error": None,
                "retrieved": retrieved or context,
            }

        answer = self._llm_service.generate_response(context=context, question=question)
        if answer is None:
            return {
                "answer": (
                    "Извините, возникла техническая ошибка при обработке запроса. "
                    "Попробуйте еще раз или обратитесь к администратору."
                ),
                "sources": sources,
                "has_context": True,
                "error": "LLM generation failed",
                "retrieved": retrieved,
            }

        return {
            "answer": answer,
            "sources": sources,
            "has_context": True,
            "error": None,
            "retrieved": retrieved,
        }

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[str, list[dict[str, Any]], str]:
        return self._retrieval_service.retrieve_context(question, top_k=top_k)
