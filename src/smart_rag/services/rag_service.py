from __future__ import annotations

import inspect
from typing import Any

from ..config import Settings, get_settings
from ..schemas.requests import QuestionTestItem, RetrievalOverrides
from .agentic_rag_service import AgenticRAGService
from .llm_service import LLMService
from .retrieval_service import RetrievalService


class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        settings: Settings | None = None,
        agentic_rag_service: AgenticRAGService | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._llm_service = llm_service
        self._settings = settings or get_settings()
        self._agentic_rag_service = agentic_rag_service
        self._mode_handlers = {
            "legacy": self._ask_detailed_legacy,
            "agentic": self._ask_detailed_agentic,
        }

    def _resolve_rag_mode(self, rag_mode: str | None) -> str:
        mode = (rag_mode or self._settings.rag_default_mode).lower()
        return mode if mode in self._mode_handlers else "legacy"

    def ask(self, question: str, top_k: int | None = None, rag_mode: str | None = None, generation_profile: str | None = None) -> dict[str, Any]:
        result = self.ask_detailed(
            question=question,
            top_k=top_k,
            include_answer=True,
            include_context=False,
            include_retrieved_text=True,
            rag_mode=rag_mode,
            generation_profile=generation_profile,
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "has_context": result["has_context"],
            "error": result["error"],
            "retrieved": result["retrieved"],
            "execution_mode": result["rag_mode"],
        }

    def _apply_reranking(
        self,
        question: str,
        analysis: dict[str, Any],
        retrieval_overrides: RetrievalOverrides | None,
        rerank_top_n: int | None = None,
    ) -> dict[str, Any]:
        hits = analysis.get("hit_details") or []
        if not hits or not analysis.get("diagnostics", {}).get("accepted"):
            return analysis
        top_n = rerank_top_n or self._settings.test_rerank_top_n
        order = self._llm_service.rerank_hits(question=question, hits=hits, top_n=top_n)
        if not order:
            return analysis
        top_slice = hits[: min(len(hits), top_n)]
        by_id = {idx: item for idx, item in enumerate(top_slice, start=1)}
        reranked_hits = [by_id[item_id] for item_id in order if item_id in by_id]
        reranked_hits.extend(hits[min(len(hits), top_n) :])
        runtime = self._retrieval_service._resolve_runtime_settings(retrieval_overrides)
        keep_n = self._settings.test_rerank_keep_n or int(runtime["max_keep"])
        keep_n = max(int(runtime["min_keep"]), keep_n)
        keep_n = min(keep_n, len(reranked_hits))
        selected = reranked_hits[:keep_n]
        selected_texts = {item["text"] for item in selected}
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        retrieved_lines: list[str] = []
        for rank, item in enumerate(reranked_hits, start=1):
            item["rank"] = rank
            item["selected"] = item["text"] in selected_texts
            item["reranked"] = True
        for item in selected:
            metadata = item.get("metadata") or {}
            context_parts.append(item["text"])
            sources.append({**metadata, "distance": item.get("distance"), "matched_query": item.get("matched_query")})
            retrieved_lines.append(
                f"Раздел: {item.get('section', 'Без раздела')}\n"
                f"(distance={float(item.get('distance') or 0):.4f}; reranked=true)\n{item['text']}"
            )
        analysis["hit_details"] = reranked_hits
        analysis["context"] = "\n\n---\n\n".join(context_parts)
        analysis["sources"] = sources
        analysis["retrieved"] = "\n\n---\n\n".join(retrieved_lines)
        analysis.setdefault("diagnostics", {})["selected_hits"] = keep_n
        analysis["diagnostics"]["reranking_applied"] = True
        analysis["diagnostics"]["rerank_top_n"] = top_n
        return analysis

    def _build_empty_detailed_response(
        self,
        question: str,
        top_k: int | None,
        retrieval_overrides: RetrievalOverrides | None,
    ) -> dict[str, Any]:
        return {
            "query": question,
            "answer": "Пожалуйста, задайте вопрос.",
            "has_context": False,
            "error": None,
            "sources": [],
            "context": None,
            "retrieved": None,
            "retrieval_diagnostics": {
                "accepted": False,
                "reason": "empty_query",
                "query_word_count": 0,
                "requested_top_k": top_k or self._settings.top_k_results,
                "top_hit_distance": None,
                "second_hit_distance": None,
                "gap12": None,
                "cutoff_distance": None,
                "total_hits": 0,
                "selected_hits": 0,
                "thresholds": self._retrieval_service._resolve_runtime_settings(retrieval_overrides),
                "retrieval_scope": "knowledge_base",
            },
            "hit_details": [],
            "query_expansion": {"enabled": False, "variants": [question]},
            "reranking": {"enabled": False, "applied": False},
            "rag_mode": self._resolve_rag_mode(None),
        }

    def _ask_detailed_legacy(self, **kwargs: Any) -> dict[str, Any]:
        question = kwargs["question"]
        top_k = kwargs.get("top_k")
        retrieval_overrides = kwargs.get("retrieval_overrides")
        include_answer = kwargs.get("include_answer", True)
        include_context = kwargs.get("include_context", True)
        include_retrieved_text = kwargs.get("include_retrieved_text", True)
        generation_profile = kwargs.get("generation_profile")
        enable_query_expansion = kwargs.get("enable_query_expansion", False)
        enable_reranking = kwargs.get("enable_reranking", False)
        query_expansion_variants = kwargs.get("query_expansion_variants")
        rerank_top_n = kwargs.get("rerank_top_n")
        vector_store_override = kwargs.get("vector_store_override")
        retrieval_scope = kwargs.get("retrieval_scope", "knowledge_base")
        retrieval_mode = kwargs.get("retrieval_mode", "smart")

        query_variants = [question]
        if enable_query_expansion and hasattr(self._llm_service, "expand_query"):
            query_variants = self._llm_service.expand_query(
                question=question,
                max_variants=query_expansion_variants or self._settings.test_query_expansion_variants,
            )
        retrieval_kwargs = {
            "query": question,
            "top_k": top_k,
            "overrides": retrieval_overrides,
            "query_variants": query_variants,
            "vector_store_override": vector_store_override,
            "retrieval_scope": retrieval_scope,
            "retrieval_mode": retrieval_mode,
        }
        try:
            signature = inspect.signature(self._retrieval_service.analyze_retrieval)
            retrieval_kwargs = {
                key: value for key, value in retrieval_kwargs.items() if key in signature.parameters
            }
        except (TypeError, ValueError):
            pass
        analysis = self._retrieval_service.analyze_retrieval(**retrieval_kwargs)
        if enable_reranking and hasattr(self._llm_service, "rerank_hits"):
            analysis = self._apply_reranking(
                question=question,
                analysis=analysis,
                retrieval_overrides=retrieval_overrides,
                rerank_top_n=rerank_top_n,
            )
        context = analysis["context"]
        sources = analysis["sources"]
        retrieved = analysis["retrieved"]
        has_context = bool(context) and "Нет релевантной информации" not in context and bool(sources)
        answer: str | None = None
        error: str | None = None
        if include_answer and has_context:
            generation_kwargs = {
                "context": context,
                "question": question,
                "sources": sources,
                "hit_details": analysis.get("hit_details"),
                "generation_profile": generation_profile,
            }
            try:
                llm_signature = inspect.signature(self._llm_service.generate_response)
                generation_kwargs = {
                    key: value for key, value in generation_kwargs.items() if key in llm_signature.parameters
                }
            except (TypeError, ValueError):
                pass
            answer = self._llm_service.generate_response(**generation_kwargs)
            if answer is None:
                error = "LLM generation failed"
                answer = "Извините, возникла техническая ошибка при обработке запроса. Попробуйте еще раз или обратитесь к администратору."
        elif include_answer and not has_context:
            answer = "В найденном контексте нет надежного ответа на этот вопрос. Проверьте формулировку запроса или обратитесь в техподдержку."
        return {
            "query": question,
            "answer": answer if include_answer else None,
            "has_context": has_context,
            "error": error,
            "sources": sources,
            "context": context if include_context else None,
            "retrieved": retrieved if include_retrieved_text else None,
            "retrieval_diagnostics": analysis["diagnostics"],
            "hit_details": analysis["hit_details"],
            "query_expansion": {"enabled": enable_query_expansion, "variants": query_variants},
            "reranking": {
                "enabled": enable_reranking,
                "applied": bool(analysis.get("diagnostics", {}).get("reranking_applied")),
            },
            "rag_mode": "legacy",
        }

    def _ask_detailed_agentic(self, **kwargs: Any) -> dict[str, Any]:
        if self._agentic_rag_service is None:
            raise RuntimeError("Agentic RAG mode is not configured")
        try:
            signature = inspect.signature(self._agentic_rag_service.run)
            supported_keys = set(signature.parameters)
            filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in supported_keys
            }
        except (TypeError, ValueError):
            filtered_kwargs = kwargs
        return self._agentic_rag_service.run(**filtered_kwargs)

    def retrieve_details(
        self,
        question: str,
        top_k: int | None = None,
        retrieval_overrides: RetrievalOverrides | None = None,
        include_context: bool = True,
        include_retrieved_text: bool = True,
        retrieval_mode: str = "smart",
        vector_store_override: Any | None = None,
        retrieval_scope: str = "knowledge_base",
    ) -> dict[str, Any]:
        normalized = (question or "").strip()
        if not normalized:
            return self._build_empty_detailed_response(question=question, top_k=top_k, retrieval_overrides=retrieval_overrides)
        analysis = self._retrieval_service.analyze_retrieval(
            query=normalized,
            top_k=top_k,
            overrides=retrieval_overrides,
            vector_store_override=vector_store_override,
            retrieval_scope=retrieval_scope,
            retrieval_mode=retrieval_mode,
        )
        has_context = bool(analysis["diagnostics"].get("accepted")) and bool(analysis.get("sources"))
        return {
            "query": normalized,
            "answer": None,
            "has_context": has_context,
            "error": None,
            "sources": analysis["sources"],
            "context": analysis["context"] if include_context else None,
            "retrieved": analysis["retrieved"] if include_retrieved_text else None,
            "retrieval_diagnostics": analysis["diagnostics"],
            "hit_details": analysis["hit_details"],
            "query_expansion": {"enabled": False, "variants": [normalized]},
            "reranking": {"enabled": False, "applied": False},
            "rag_mode": self._resolve_rag_mode(None),
        }

    def ask_detailed(
        self,
        question: str,
        top_k: int | None = None,
        retrieval_overrides: RetrievalOverrides | None = None,
        include_answer: bool = True,
        include_context: bool = True,
        include_retrieved_text: bool = True,
        generation_profile: str | None = None,
        enable_query_expansion: bool = False,
        enable_reranking: bool = False,
        query_expansion_variants: int | None = None,
        rerank_top_n: int | None = None,
        rag_mode: str | None = None,
        vector_store_override: Any | None = None,
        retrieval_scope: str = "knowledge_base",
        retrieval_mode: str = "smart",
    ) -> dict[str, Any]:
        normalized = (question or "").strip()
        if not normalized:
            return self._build_empty_detailed_response(question=question, top_k=top_k, retrieval_overrides=retrieval_overrides)
        mode = self._resolve_rag_mode(rag_mode)
        handler = self._mode_handlers[mode]
        kwargs = {
            "question": normalized,
            "top_k": top_k,
            "retrieval_overrides": retrieval_overrides,
            "include_answer": include_answer,
            "include_context": include_context,
            "include_retrieved_text": include_retrieved_text,
            "generation_profile": generation_profile,
            "enable_query_expansion": enable_query_expansion,
            "enable_reranking": enable_reranking,
            "query_expansion_variants": query_expansion_variants,
            "rerank_top_n": rerank_top_n,
            "vector_store_override": vector_store_override,
            "retrieval_scope": retrieval_scope,
            "retrieval_mode": retrieval_mode,
        }
        try:
            signature = inspect.signature(handler)
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())}
        except (TypeError, ValueError):
            pass
        result = handler(**kwargs)
        result["rag_mode"] = mode
        return result

    def run_test_queries(
        self,
        items: list[QuestionTestItem],
        top_k: int | None = None,
        retrieval_overrides: RetrievalOverrides | None = None,
        include_answers: bool = True,
        include_context: bool = True,
        include_retrieved_text: bool = True,
        enable_query_expansion: bool = True,
        enable_reranking: bool = True,
        query_expansion_variants: int | None = None,
        generation_profile: str | None = None,
        rerank_top_n: int | None = None,
        rag_mode: str | None = None,
        vector_store_override: Any | None = None,
        retrieval_scope: str = "knowledge_base",
        retrieval_mode: str = "smart",
        uploaded_document_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        answered_count = 0
        context_count = 0
        error_count = 0
        for item in items:
            response = self.ask_detailed(
                question=item.query,
                top_k=top_k,
                retrieval_overrides=retrieval_overrides,
                include_answer=include_answers,
                include_context=include_context,
                include_retrieved_text=include_retrieved_text,
                enable_query_expansion=enable_query_expansion and self._settings.test_enable_query_expansion,
                enable_reranking=enable_reranking and self._settings.test_enable_reranking,
                generation_profile=generation_profile,
                query_expansion_variants=query_expansion_variants,
                rerank_top_n=rerank_top_n,
                rag_mode=rag_mode,
                vector_store_override=vector_store_override,
                retrieval_scope=retrieval_scope,
                retrieval_mode=retrieval_mode,
            )
            result_item = {
                "question_id": item.question_id,
                "query": item.query,
                "metadata": item.metadata,
                **response,
            }
            results.append(result_item)
            if result_item.get("answer"):
                answered_count += 1
            if result_item.get("has_context"):
                context_count += 1
            if result_item.get("error"):
                error_count += 1
        response = {
            "summary": {
                "total_questions": len(items),
                "answered_questions": answered_count,
                "questions_with_context": context_count,
                "questions_without_context": len(items) - context_count,
                "questions_with_error": error_count,
            },
            "effective_params": {
                "top_k": top_k or self._settings.top_k_results,
                "retrieval_overrides": retrieval_overrides.model_dump(exclude_none=True) if retrieval_overrides else None,
                "rag_mode": self._resolve_rag_mode(rag_mode),
                "generation_profile": generation_profile or self._settings.generation_profile,
                "retrieval_scope": retrieval_scope,
                "retrieval_mode": retrieval_mode,
                "embedding_provider": self._settings.embedding_provider,
                "embedding_model": self._settings.embedding_model,
            },
            "results": results,
        }
        if uploaded_document_info:
            response["effective_params"]["uploaded_document"] = uploaded_document_info
        return response
