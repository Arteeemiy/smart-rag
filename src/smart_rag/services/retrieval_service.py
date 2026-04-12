from __future__ import annotations

import logging
from collections import OrderedDict
import re
from typing import Any

from ..clients.vector_store_client import VectorStoreClient
from ..config import Settings
from ..schemas.requests import RetrievalOverrides

logger = logging.getLogger(__name__)


class RetrievalService:
    _ALIAS_MAP: dict[str, list[str]] = {
        "бмс": ["bms", "лк ук", "личный кабинет управляющей организации", "система диспетчеризации здания"],
        "bms": ["бмс", "лк ук", "личный кабинет управляющей организации", "система диспетчеризации здания"],
        "лк ук": ["личный кабинет управляющей организации", "личный кабинет управляющей компании", "bms", "бмс"],
        "скуд": ["система контроля и управления доступом", "скуд"],
        "сла": ["sla", "соглашение об уровне обслуживания"],
        "sla": ["сла", "соглашение об уровне обслуживания"],
        "едс": ["единая диспетчерская служба", "едс"],
        "тмц": ["товарно-материальные ценности", "тмц"],
        "резидент": ["резидент", "пользователь ujin", "справочник резидентов"],
        "кабинет ук": ["лк ук", "личный кабинет управляющей компании", "bms"],
        "кабинет управляющей компании": ["лк ук", "личный кабинет управляющей компании", "bms"],
    }

    _ANCHOR_TERMS: dict[str, tuple[str, ...]] = {
        "address": ("адрес", "ссылка", "url", "https", "bms.ujin.tech"),
        "memory": ("оперативная память", "озу", "ram", "гб"),
        "browser": ("браузер", "chrome", "google chrome"),
        "os": ("операционная система", "windows", "macos", "ubuntu", "linux"),
        "dashboard": ("дашборд", "виджет", "сигнал", "архив", "индикац", "звук", "восклицан"),
        "profile": ("профиль", "пароль", "аватар", "фото", "логин"),
        "support": ("техподдерж", "инструкция", "руководств", "помощ"),
        "objects": ("здание", "комплекс", "объект", "подъезд", "этаж", "помещени", "парковк"),
        "registry": ("фильтр", "сортировк", "колонк", "реестр", "видимост"),
        "auth": ("вход", "авториз", "зайти", "выйти", "разлогин", "доступ"),
    }

    _INTENT_RULES: dict[str, dict[str, tuple[str, ...]]] = {
        "support_contact": {
            "query": ("техподдерж", "тп", "поддержк", "о системе", "задать вопрос", "связаться"),
            "section": ("подготовка к работе",),
            "subsection": ("техподдержка",),
            "title": ("техподдержка", "о системе"),
            "record_type": ("instruction", "procedure", "ui_navigation", "fact"),
        },
        "instruction_lookup": {
            "query": ("инструкция", "руководств", "как пользоваться", "где почитать", "пользователь", "мануал"),
            "section": ("подготовка к работе",),
            "subsection": ("инструкция",),
            "title": ("инструкция", "руководство пользователя"),
            "record_type": ("instruction", "procedure", "ui_navigation", "fact"),
        },
        "glossary": {
            "query": ("что такое", "кто такой", "что за", "что означает"),
            "section": ("список терминов и сокращений",),
            "subsection": (),
            "title": ("список терминов и сокращений",),
            "record_type": ("fact", "definition", "faq"),
        },
        "requirements": {
            "query": ("оперативн", "озу", "ram", "браузер", "операцион", "windows", "macos", "linux", "ubuntu", "технические требования"),
            "section": ("назначение и условия применения",),
            "subsection": (),
            "title": ("назначение и условия применения",),
            "record_type": ("fact",),
        },
        "prep": {
            "query": ("адрес", "ссылка", "вход", "авториз", "зайти", "инструкция", "руководств", "техподдерж", "профиль", "пароль", "фото", "выйти", "разлогин"),
            "section": ("подготовка к работе",),
            "subsection": ("вход в лк ук", "профиль", "инструкция", "техподдержка", "выход из лк ук"),
            "title": ("вход в лк ук", "профиль", "инструкция", "техподдержка", "выход из лк ук"),
            "record_type": ("ui_navigation", "fact", "procedure", "instruction"),
        },
        "registry": {
            "query": ("фильтр", "сортировк", "колонк", "реестр", "видимост", "ширин"),
            "section": ("основные операции",),
            "subsection": ("настройка реестра",),
            "title": ("настройка реестра", "сортировка реестра", "фильтрация реестра"),
            "record_type": ("procedure", "instruction", "fact"),
        },
        "dashboard_alert": {
            "query": ("восклиц", "!", "знак", "иконка", "звук", "оповещ"),
            "section": ("работа с системой",),
            "subsection": ("дашборд",),
            "title": ("режим активных сигналов", "дашборд", "показатели объекта и их отклонения"),
            "record_type": ("fact", "procedure", "instruction"),
        },
        "dashboard_archive": {
            "query": ("архив", "архивир", "архивный", "из архива", "в архив", "вернуть сигнал", "достать сигнал", "архив сигналов"),
            "section": ("работа с системой",),
            "subsection": ("дашборд",),
            "title": ("режим архивных сигналов", "режим активных сигналов", "дашборд"),
            "record_type": ("fact", "procedure", "instruction"),
        },
        "dashboard": {
            "query": ("дашборд", "виджет", "архив", "сигнал", "звук", "восклиц", "лифтов"),
            "section": ("работа с системой",),
            "subsection": ("дашборд",),
            "title": ("дашборд", "режим активных сигналов", "режим архивных сигналов", "настройка виджетов для дашбордов", "как работает дашборд?", "цветовая индикация счетчика по порогам", "устройства офлайн"),
            "record_type": ("fact", "procedure", "instruction"),
        },
        "menu_behavior": {
            "query": ("меню", "боковое меню", "главное меню", "развернут", "свернуть", "зафиксир", "не сворачив", "закреп"),
            "section": ("основные операции",),
            "subsection": ("главное меню",),
            "title": ("главное меню",),
            "record_type": ("instruction", "ui_navigation", "procedure"),
        },
        "objects": {
            "query": ("здание", "комплекс", "объект", "подъезд", "этаж", "помещени", "парковк", "гостев", "верификац", "квартир"),
            "section": ("работа с системой",),
            "subsection": ("справочники", "настройки"),
            "title": ("справочник объектов (new)", "работа с деревом объектов", "работа с карточкой комплекса", "работа с карточкой здания", "работа с карточками здания и парковки", "работа с карточкой подъезда", "работа с карточкой этажа", "работа с карточкой помещения", "настройка парковок"),
            "record_type": ("fact", "procedure", "instruction", "ui_navigation"),
        },
    }

    def __init__(self, vector_store_client: VectorStoreClient, settings: Settings) -> None:
        self._vector_store_client = vector_store_client
        self._settings = settings

    @staticmethod
    def _tokenize(text: str | None) -> set[str]:
        return {token for token in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE) if len(token) > 1}

    def _classify_intent(self, query: str) -> str:
        lowered = (query or "").lower()
        for intent, spec in self._INTENT_RULES.items():
            if any(marker in lowered for marker in spec["query"]):
                return intent
        return "general"

    def _expand_alias_variants(self, query: str) -> list[str]:
        normalized = (query or "").strip()
        if not normalized:
            return []
        lowered = normalized.lower()
        variants: list[str] = []
        for alias, replacements in self._ALIAS_MAP.items():
            if alias in lowered:
                for replacement in replacements:
                    candidate = re.sub(re.escape(alias), replacement, lowered, flags=re.IGNORECASE)
                    if candidate.strip() and candidate.strip() != lowered:
                        variants.append(candidate.strip())
        return variants

    def _build_intent_variants(self, query: str, intent: str) -> list[str]:
        lowered = (query or "").strip().lower()
        variants: list[str] = []
        if intent == "glossary":
            variants.extend([
                f"список терминов и сокращений {lowered}",
                f"термин {lowered}",
            ])
        elif intent == "requirements":
            variants.extend([
                f"назначение и условия применения {lowered}",
                f"технические требования {lowered}",
            ])
        elif intent == "prep":
            variants.extend([
                f"подготовка к работе {lowered}",
                f"вход в лк ук {lowered}",
                f"инструкция {lowered}",
            ])
        elif intent == "instruction_lookup":
            variants.extend([
                f"инструкция {lowered}",
                f"руководство пользователя {lowered}",
                f"подготовка к работе {lowered}",
                f"инструкция лк ук {lowered}",
                f"инструкция bms {lowered}",
            ])
        elif intent == "support_contact":
            variants.extend([
                f"техподдержка {lowered}",
                f"о системе {lowered}",
                f"как обратиться в техподдержку {lowered}",
            ])
        elif intent == "registry":
            variants.extend([
                f"настройка реестра {lowered}",
                f"основные операции {lowered}",
            ])
        elif intent == "dashboard_alert":
            variants.extend([
                f"режим активных сигналов {lowered}",
                f"дашборд {lowered}",
                f"знак восклицания виджет {lowered}",
            ])
        elif intent == "dashboard_archive":
            variants.extend([
                f"режим архивных сигналов {lowered}",
                f"архив сигналов {lowered}",
                f"дашборд архив {lowered}",
            ])
        elif intent == "dashboard":
            variants.extend([
                f"дашборд {lowered}",
                f"режим активных сигналов {lowered}",
                f"режим архивных сигналов {lowered}",
            ])
        elif intent == "menu_behavior":
            variants.extend([
                f"главное меню {lowered}",
                f"боковая панель {lowered}",
            ])
        elif intent == "objects":
            variants.extend([
                f"справочник объектов {lowered}",
                f"работа с карточкой здания {lowered}",
                f"работа с карточкой комплекса {lowered}",
                f"работа с карточкой помещения {lowered}",
            ])
        return [item for item in variants if item and item != lowered]

    def _resolve_runtime_settings(self, overrides: RetrievalOverrides | None) -> dict[str, float | int]:
        return {
            "gate_strong": overrides.gate_strong if overrides and overrides.gate_strong is not None else self._settings.gate_strong,
            "gate_max": overrides.gate_max if overrides and overrides.gate_max is not None else self._settings.gate_max,
            "gap_min": overrides.gap_min if overrides and overrides.gap_min is not None else self._settings.gap_min,
            "window": overrides.window if overrides and overrides.window is not None else self._settings.window,
            "min_keep": overrides.min_keep if overrides and overrides.min_keep is not None else self._settings.min_keep,
            "max_keep": overrides.max_keep if overrides and overrides.max_keep is not None else self._settings.max_keep,
        }

    def _search_hits(
        self,
        query: str,
        n_results: int,
        query_variants: list[str] | None = None,
        vector_store_override: Any | None = None,
    ):
        intent = self._classify_intent(query)
        variants = [item.strip() for item in (query_variants or [query]) if item and item.strip()]
        for alias_variant in self._expand_alias_variants(query):
            if alias_variant not in variants:
                variants.append(alias_variant)
        for intent_variant in self._build_intent_variants(query, intent):
            if intent_variant not in variants:
                variants.append(intent_variant)
        if query not in variants:
            variants.insert(0, query)
        aggregated: OrderedDict[tuple[str, str], tuple[Any, float, str]] = OrderedDict()
        query_cap = max(n_results, 8)
        for variant in variants:
            hits = self._vector_store_client.similarity_search_with_score(variant, k=query_cap, vector_store=vector_store_override)
            for doc, score in hits:
                metadata = doc.metadata or {}
                key = (str(metadata.get("record_id") or ""), doc.page_content)
                current = aggregated.get(key)
                if current is None or float(score) < current[1]:
                    aggregated[key] = (doc, float(score), variant)
        merged = list(aggregated.values())
        merged.sort(key=lambda item: item[1])
        return merged[: max(query_cap, len(variants) * 2)]

    @staticmethod
    def _build_source_item(metadata: dict[str, Any], distance: float, matched_query: str) -> dict[str, Any]:
        return {**metadata, "distance": distance, "matched_query": matched_query}

    @staticmethod
    def _build_retrieved_preview(doc_text: str, metadata: dict[str, Any], distance: float, matched_query: str) -> str:
        lines = [
            f"Раздел: {metadata.get('section') or 'Без раздела'}",
            f"Подраздел: {metadata.get('subsection') or '—'}",
            f"Тип записи: {metadata.get('record_type') or 'unknown'}",
            f"Заголовок: {metadata.get('title') or '—'}",
            f"distance={distance:.4f}; matched_query={matched_query}",
        ]
        question = metadata.get("question")
        if question:
            lines.append(f"Вопрос: {question}")
        lines.append(doc_text)
        return "\n".join(lines)

    def _intent_bonus(self, query: str, metadata: dict[str, Any]) -> float:
        intent = self._classify_intent(query)
        if intent == "general":
            return 0.0
        spec = self._INTENT_RULES[intent]
        title = str(metadata.get("title") or "").lower()
        section = str(metadata.get("section") or "").lower()
        subsection = str(metadata.get("subsection") or "").lower()
        record_type = str(metadata.get("record_type") or "").lower()
        bonus = 0.0
        if any(marker in section for marker in spec["section"]):
            bonus += 0.35
        if any(marker in subsection for marker in spec["subsection"]):
            bonus += 0.35
        if any(marker in title for marker in spec["title"]):
            bonus += 0.42
        if record_type in spec["record_type"]:
            bonus += 0.08
        if intent == "dashboard" and metadata.get("section") == "Работа с системой" and subsection == "дашборд":
            bonus += 0.15
        if intent == "prep" and section == "подготовка к работе":
            bonus += 0.15
        if intent == "support_contact" and section == "подготовка к работе":
            bonus += 0.28
            if "техподдержка" in title or "о системе" in title or "техподдержка" in subsection:
                bonus += 0.32
        if intent == "instruction_lookup" and section == "подготовка к работе":
            bonus += 0.22
            if "инструкция" in title or "руководство" in title:
                bonus += 0.22
        if intent == "dashboard_alert" and section == "работа с системой":
            bonus += 0.18
            if any(marker in title for marker in ["режим активных сигналов", "показатели объекта и их отклонения"]):
                bonus += 0.24
        if intent == "dashboard_archive" and section == "работа с системой":
            bonus += 0.2
            if "режим архивных сигналов" in title:
                bonus += 0.34
            if "режим архивных сигналов" in subsection:
                bonus += 0.28
        if intent == "menu_behavior" and section == "основные операции":
            bonus += 0.2
            if "главное меню" in title or "главное меню" in subsection:
                bonus += 0.3
        if intent == "objects" and any(marker in (query or "").lower() for marker in ["квартир"]):
            if any(marker in title for marker in ["карточкой помещения", "карточкой здания", "справочник объектов"]):
                bonus += 0.18
        if intent == "registry" and section == "основные операции":
            bonus += 0.15
        return bonus

    def _lexical_bonus(self, query: str, metadata: dict[str, Any], doc_text: str) -> float:
        query_tokens = self._tokenize(query)
        lowered_query = (query or "").lower()
        if not query_tokens:
            return 0.0
        haystack_parts = [
            metadata.get("title"),
            metadata.get("question"),
            metadata.get("keywords"),
            metadata.get("aliases"),
            metadata.get("section_path"),
            metadata.get("record_type"),
            metadata.get("section"),
            metadata.get("subsection"),
            doc_text[:2200],
        ]
        haystack_tokens: set[str] = set()
        haystack_text = " ".join(str(part) for part in haystack_parts if part is not None).lower()
        for part in haystack_parts:
            haystack_tokens.update(self._tokenize(str(part) if part is not None else ""))
        if not haystack_tokens:
            return 0.0
        overlap = len(query_tokens & haystack_tokens) / max(len(query_tokens), 1)
        bonus = overlap * 0.9
        bonus += self._intent_bonus(query, metadata)
        if metadata.get("chunking_hint") == "atomic":
            bonus += 0.15
        if metadata.get("priority") == "high":
            bonus += 0.08
        title_text = str(metadata.get("title") or "").lower()
        section_text = str(metadata.get("section") or "").lower()
        subsection_text = str(metadata.get("subsection") or "").lower()
        if any(token in lowered_query for token in ["что такое", "кто такой", "что за", "что означает"]):
            if any(marker in section_text for marker in ["список терминов", "назначение и условия применения"]):
                bonus += 0.25
            if any(marker in title_text for marker in ["список терминов", "назначение и условия применения"]):
                bonus += 0.22
        if any(token in lowered_query for token in ["как ", "где ", "открыть", "добавить", "удалить", "перейти", "настроить", "редактировать"]):
            if metadata.get("record_type") in {"procedure", "instruction", "ui_navigation", "fact"}:
                bonus += 0.16
        for group_name, anchors in self._ANCHOR_TERMS.items():
            if any(anchor in lowered_query for anchor in anchors):
                hit_count = sum(1 for anchor in anchors if anchor in haystack_text)
                if hit_count:
                    bonus += min(0.32, 0.06 * hit_count)
                if group_name == "address" and ("https://" in haystack_text or "bms.ujin.tech" in haystack_text):
                    bonus += 0.45
                if group_name == "memory" and "оперативная память" in haystack_text and "гб" in haystack_text:
                    bonus += 0.38
                if group_name == "browser" and "chrome" in haystack_text:
                    bonus += 0.28
                if group_name == "os" and any(token in haystack_text for token in ["windows", "macos", "ubuntu", "linux mint", "linux"]):
                    bonus += 0.3
                if group_name == "support" and any(token in title_text for token in ["инструкция", "техподдержка", "вход в лк ук", "выход из лк ук"]):
                    bonus += 0.22
        if any(token in lowered_query for token in ["дашборд", "виджет", "архив", "сигнал"]):
            if any(marker in subsection_text for marker in ["дашборд"]) or any(marker in title_text for marker in ["режим активных сигналов", "режим архивных сигналов", "как работает дашборд", "настройка виджетов"]):
                bonus += 0.24
        intent = self._classify_intent(query)
        if intent == "instruction_lookup" and not any(marker in haystack_text for marker in ["инструкция", "руководство пользователя"]):
            bonus -= 0.22
        if intent == "support_contact" and not any(marker in haystack_text for marker in ["техподдержка", "о системе", "службы технической поддержки"]):
            bonus -= 0.24
        if intent == "dashboard_archive":
            if any(marker in haystack_text for marker in ["фильтр", "настройка реестра", "колонк"]) and "режим архивных сигналов" not in haystack_text:
                bonus -= 0.4
            if "режим архивных сигналов" in haystack_text:
                bonus += 0.26
        if intent == "dashboard_alert":
            if any(marker in haystack_text for marker in ["режим активных сигналов", "показатели объекта и их отклонения"]):
                bonus += 0.22
            if any(marker in haystack_text for marker in ["фильтр", "архив квитанций"]) and "режим активных сигналов" not in haystack_text:
                bonus -= 0.25
        if intent == "menu_behavior" and "главное меню" not in haystack_text:
            bonus -= 0.25
        if "едс" in lowered_query and "единая диспетчерская служба" in haystack_text:
            bonus += 0.5
        if any(marker in lowered_query for marker in ["оператив", "памят", "озу", "ram"]) and all(marker in haystack_text for marker in ["оперативная память", "гб"]):
            bonus += 0.45
        if any(marker in lowered_query for marker in ["квартир", "квартиры"]) and any(marker in haystack_text for marker in ["помещени", "карточкой помещения", "справочник объектов"]):
            bonus += 0.18
        return bonus

    def _deduplicate_hits(self, hits: list[tuple[Any, float, str]]) -> list[tuple[Any, float, str]]:
        deduped: list[tuple[Any, float, str]] = []
        seen: set[tuple[str, str]] = set()
        for doc, score, matched_query in hits:
            metadata = doc.metadata or {}
            key = (
                str(metadata.get("record_id") or ""),
                str(metadata.get("section_path") or metadata.get("title") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append((doc, score, matched_query))
        return deduped

    def _rerank_hits(self, query: str, hits: list[tuple[Any, float, str]], strategy: str) -> list[tuple[Any, float, str]]:
        if strategy == "vector":
            return self._deduplicate_hits(hits)
        rescored: list[tuple[Any, float, str]] = []
        for doc, score, matched_query in hits:
            metadata = doc.metadata or {}
            adjusted = float(score) - self._lexical_bonus(query, metadata, doc.page_content)
            rescored.append((doc, adjusted, matched_query))
        rescored.sort(key=lambda item: item[1])
        return self._deduplicate_hits(rescored)

    def _select_coherent_hits(self, query: str, hits: list[tuple[Any, float, str]], runtime: dict[str, float | int]) -> list[tuple[Any, float, str]]:
        """Prefer supporting chunks from the same logical branch for non-factoid questions."""
        if not hits:
            return []
        lowered = (query or '').lower()
        is_factoid = any(token in lowered for token in [
            'ссылка', 'адрес', 'url', 'сколько', 'браузер', 'операцион', 'оператив', 'озу', 'ram', 'что такое', 'кто такой', 'что за'
        ]) or len(lowered.split()) <= 4
        d1 = float(hits[0][1])
        cutoff = d1 + float(runtime['window'])
        selected = [(doc, float(score), matched_query) for doc, score, matched_query in hits if float(score) <= cutoff]
        if len(selected) < int(runtime['min_keep']):
            selected = [(doc, float(score), matched_query) for doc, score, matched_query in hits[: int(runtime['min_keep'])]]
        if is_factoid:
            return selected[:1]

        primary_doc, _, _ = selected[0]
        primary_meta = primary_doc.metadata or {}
        primary_section = str(primary_meta.get('section') or '').strip().lower()
        primary_subsection = str(primary_meta.get('subsection') or '').strip().lower()
        primary_title = str(primary_meta.get('title') or '').strip().lower()
        coherent: list[tuple[Any, float, str]] = [selected[0]]
        fallback: list[tuple[Any, float, str]] = []
        for item in selected[1:]:
            doc, score, matched_query = item
            meta = doc.metadata or {}
            section = str(meta.get('section') or '').strip().lower()
            subsection = str(meta.get('subsection') or '').strip().lower()
            title = str(meta.get('title') or '').strip().lower()
            same_title = primary_title and title and title == primary_title
            same_subsection = primary_subsection and subsection and subsection == primary_subsection
            same_section = primary_section and section and section == primary_section
            if same_title or same_subsection or same_section:
                coherent.append(item)
            else:
                fallback.append(item)
        limit = int(runtime['max_keep'])
        if len(coherent) < int(runtime['min_keep']):
            coherent.extend(fallback[: max(0, int(runtime['min_keep']) - len(coherent))])
        elif len(coherent) < min(3, limit):
            coherent.extend(fallback[: max(0, min(3, limit) - len(coherent))])
        return coherent[:limit]

    def analyze_retrieval(
        self,
        query: str,
        top_k: int | None = None,
        overrides: RetrievalOverrides | None = None,
        query_variants: list[str] | None = None,
        vector_store_override: Any | None = None,
        retrieval_scope: str = "knowledge_base",
        retrieval_mode: str = "smart",
    ) -> dict[str, Any]:
        words = [word for word in (query or "").strip().split() if word]
        n_results = top_k or self._settings.top_k_results
        runtime = self._resolve_runtime_settings(overrides)
        is_short = len(words) <= 3
        try:
            raw_hits = self._search_hits(query=query, n_results=n_results, query_variants=query_variants, vector_store_override=vector_store_override)
            hits = self._rerank_hits(query=query, hits=raw_hits, strategy=retrieval_mode)
            if not hits:
                return {
                    "context": "Нет релевантной информации в базе знаний.",
                    "sources": [],
                    "retrieved": "",
                    "hit_details": [],
                    "diagnostics": {
                        "accepted": False,
                        "reason": "no_hits",
                        "query_word_count": len(words),
                        "requested_top_k": n_results,
                        "top_hit_distance": None,
                        "second_hit_distance": None,
                        "gap12": None,
                        "cutoff_distance": None,
                        "total_hits": 0,
                        "selected_hits": 0,
                        "thresholds": runtime,
                        "query_variants": query_variants or [query],
                        "retrieval_scope": retrieval_scope,
                        "retrieval_mode": retrieval_mode,
                        "intent": self._classify_intent(query),
                    },
                }
            d1 = float(hits[0][1])
            d2 = float(hits[1][1]) if len(hits) > 1 else d1
            gap12 = d2 - d1
            accepted = False
            reason = "rejected_unknown"
            if retrieval_mode == "vector":
                accepted = True
                reason = "accepted_vector_mode"
            elif is_short:
                accepted = d1 <= 11.0
                reason = "accepted_short_query" if accepted else "rejected_short_query_distance"
            elif d1 <= float(runtime["gate_strong"]):
                accepted = True
                reason = "accepted_strong_gate"
            else:
                gap_need = 1.6 if d1 > 11.0 else float(runtime["gap_min"])
                accepted = d1 <= float(runtime["gate_max"]) and gap12 >= gap_need
                reason = "accepted_gap_gate" if accepted else "rejected_gate_or_gap"

            hit_details: list[dict[str, Any]] = []
            for rank, (doc, score, matched_query) in enumerate(hits, start=1):
                metadata = doc.metadata or {}
                hit_details.append(
                    {
                        "rank": rank,
                        "distance": float(score),
                        "selected": False,
                        "metadata": metadata,
                        "record_id": metadata.get("record_id"),
                        "record_type": metadata.get("record_type"),
                        "section": metadata.get("section", "Без раздела"),
                        "subsection": metadata.get("subsection"),
                        "title": metadata.get("title"),
                        "question": metadata.get("question"),
                        "source_doc": metadata.get("source_doc"),
                        "page_start": metadata.get("page_start"),
                        "page_end": metadata.get("page_end"),
                        "source": metadata.get("source_doc") or metadata.get("source_type"),
                        "text": doc.page_content,
                        "matched_query": matched_query,
                    }
                )

            if not accepted:
                return {
                    "context": "Нет релевантной информации в базе знаний.",
                    "sources": [],
                    "retrieved": "",
                    "hit_details": hit_details,
                    "diagnostics": {
                        "accepted": False,
                        "reason": reason,
                        "query_word_count": len(words),
                        "requested_top_k": n_results,
                        "top_hit_distance": d1,
                        "second_hit_distance": d2,
                        "gap12": gap12,
                        "cutoff_distance": None,
                        "total_hits": len(hits),
                        "selected_hits": 0,
                        "thresholds": runtime,
                        "query_variants": query_variants or [query],
                        "retrieval_scope": retrieval_scope,
                        "retrieval_mode": retrieval_mode,
                        "intent": self._classify_intent(query),
                    },
                }

            cutoff = d1 + float(runtime["window"])
            picked = self._select_coherent_hits(query=query, hits=hits, runtime=runtime)
            selected_texts = {doc.page_content for doc, _, _ in picked}
            context_parts: list[str] = []
            sources: list[dict[str, Any]] = []
            retrieved_lines: list[str] = []
            for item in hit_details:
                item["selected"] = item["text"] in selected_texts
            for doc, distance, matched_query in picked:
                metadata = doc.metadata or {}
                context_parts.append(doc.page_content)
                sources.append(self._build_source_item(metadata, distance, matched_query))
                retrieved_lines.append(self._build_retrieved_preview(doc.page_content, metadata, distance, matched_query))
            return {
                "context": "\n\n---\n\n".join(context_parts),
                "sources": sources,
                "retrieved": "\n\n---\n\n".join(retrieved_lines),
                "hit_details": hit_details,
                "diagnostics": {
                    "accepted": True,
                    "reason": reason,
                    "query_word_count": len(words),
                    "requested_top_k": n_results,
                    "top_hit_distance": d1,
                    "second_hit_distance": d2,
                    "gap12": gap12,
                    "cutoff_distance": cutoff,
                    "total_hits": len(hits),
                    "selected_hits": len(picked),
                    "thresholds": runtime,
                    "query_variants": query_variants or [query],
                    "retrieval_scope": retrieval_scope,
                    "intent": self._classify_intent(query),
                },
            }
        except Exception:
            logger.exception("Retriever failed")
            return {
                "context": "",
                "sources": [],
                "retrieved": "",
                "hit_details": [],
                "diagnostics": {
                    "accepted": False,
                    "reason": "retriever_error",
                    "query_word_count": len([word for word in (query or "").strip().split() if word]),
                    "requested_top_k": top_k or self._settings.top_k_results,
                    "top_hit_distance": None,
                    "second_hit_distance": None,
                    "gap12": None,
                    "cutoff_distance": None,
                    "total_hits": 0,
                    "selected_hits": 0,
                    "thresholds": self._resolve_runtime_settings(overrides),
                    "query_variants": query_variants or [query],
                    "retrieval_scope": retrieval_scope,
                    "intent": self._classify_intent(query),
                },
            }

    def retrieve_context(self, query: str, top_k: int | None = None) -> tuple[str, list[dict[str, Any]], str]:
        analysis = self.analyze_retrieval(query=query, top_k=top_k)
        return analysis["context"], analysis["sources"], analysis["retrieved"]
