import logging
from typing import Any

from ..clients.vector_store_client import VectorStoreClient
from ..config import Settings

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self, vector_store_client: VectorStoreClient, settings: Settings
    ) -> None:
        self._vector_store_client = vector_store_client
        self._settings = settings

    def retrieve_context(
        self, query: str, top_k: int | None = None
    ) -> tuple[str, list[dict[str, Any]], str]:
        try:
            n_results = top_k or self._settings.top_k_results
            hits = self._vector_store_client.vector_store.similarity_search_with_score(
                query, k=n_results
            )

            if not hits:
                return "Нет релевантной информации в базе знаний.", [], ""

            d1 = float(hits[0][1])
            d2 = float(hits[1][1]) if len(hits) > 1 else d1
            gap12 = d2 - d1

            words = [word for word in (query or "").strip().split() if word]
            is_short = len(words) <= 2

            accepted = False
            if is_short:
                accepted = d1 <= 11.0
            elif d1 <= self._settings.gate_strong:
                accepted = True
            else:
                gap_need = 1.6 if d1 > 11.0 else self._settings.gap_min
                accepted = d1 <= self._settings.gate_max and gap12 >= gap_need

            if not accepted:
                return "Нет релевантной информации в базе знаний.", [], ""

            cutoff = d1 + self._settings.window
            picked = [
                (doc, float(score)) for doc, score in hits if float(score) <= cutoff
            ]

            if len(picked) < self._settings.min_keep:
                picked = [
                    (doc, float(score))
                    for doc, score in hits[: self._settings.min_keep]
                ]

            picked = picked[: self._settings.max_keep]

            context_parts: list[str] = []
            sources: list[dict[str, Any]] = []
            retrieved_lines: list[str] = []

            for doc, distance in picked:
                metadata = doc.metadata or {}
                section = metadata.get("section", "Без раздела")

                context_parts.append(doc.page_content)
                sources.append({**metadata, "distance": distance})

                retrieved_lines.append(
                    f"Раздел: {section}\n"
                    f"(distance={distance:.4f})\n"
                    f"{doc.page_content}"
                )

            context = "\n\n---\n\n".join(context_parts)
            retrieved = "\n\n---\n\n".join(retrieved_lines)

            return context, sources, retrieved

        except Exception:
            logger.exception("Retriever failed")
            return "", [], ""
