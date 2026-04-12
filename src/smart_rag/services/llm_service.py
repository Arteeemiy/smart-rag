from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..clients.base_llm_client import BaseLLMClient
from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMService:
    _REFUSAL_MARKERS = (
        "информации недостаточно",
        "информация недостаточно",
        "нет информации",
        "нет релевантной информации",
        "недостаточно данных",
        "в контексте нет",
        "в предоставленном контексте нет",
    )

    _URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
    _MEMORY_RE = re.compile(r"(?:оперативн\w* память|озу|ram)[^\n]{0,40}?(?:от\s*)?(\d+)\s*гб", flags=re.IGNORECASE)
    _GB_FALLBACK_RE = re.compile(r"оперативн\w* память[^\n]{0,80}?(\d+)\s*гб", flags=re.IGNORECASE)
    _OS_RE = re.compile(r"(?:ms\s*windows|windows|macos|ubuntu\s*linux|linux\s*mint)", flags=re.IGNORECASE)
    _BROWSER_RE = re.compile(r"google\s*chrome(?:[^\n]{0,40}?\d+)?", flags=re.IGNORECASE)

    def __init__(self, llm_client: BaseLLMClient, settings: Settings | None = None) -> None:
        self._llm_client = llm_client
        self._settings = settings or get_settings()

    def _log_model(self, action: str) -> None:
        logger.info(
            "%s started | provider=%s | model=%s",
            action,
            self._llm_client.provider_name,
            self._llm_client.model_name or "unknown",
        )

    def _classify_question(self, question: str) -> str:
        normalized = (question or "").strip().lower()
        if not normalized:
            return "general"
        if any(marker in normalized for marker in ["какая ссылка", "какой адрес", "адрес", "ссылка", "сколько", "на каком браузере", "какая ос", "на какой операционной системе", "какой браузер", "оперативной памяти", "озу", "ram", "какое значение", "что значит", "где находится"]):
            return "factoid"
        if any(marker in normalized for marker in ["что такое", "кто такой", "что за", "что означает", "для чего нужен", "зачем нужна", "зачем нужен"]):
            return "definition"
        if any(marker in normalized for marker in ["как ", "где ", "можно ли", "как-то", "как сделать", "как открыть", "как перейти", "как добавить", "как удалить", "как вернуть", "как достать", "как использовать", "как настроить", "как изменить", "как редактировать", "как выйти"]):
            return "procedure"
        if len(normalized.split()) <= 4:
            return "factoid"
        return "general"

    def _resolve_system_prompt(self, generation_profile: str | None, question_type: str, rescue: bool = False) -> str:
        profile = (generation_profile or self._settings.generation_profile or "grounded_v2").lower()
        if rescue:
            return self._settings.system_prompt_grounded_rescue
        if profile == "baseline":
            return self._settings.system_prompt
        if question_type == "factoid":
            return self._settings.system_prompt_grounded_factoid
        if question_type == "definition":
            return self._settings.system_prompt_grounded_definition
        if question_type == "procedure":
            return self._settings.system_prompt_grounded_procedure
        return self._settings.system_prompt_grounded_v2

    def _build_structured_context(
        self,
        context: str,
        sources: list[dict[str, Any]] | None = None,
        hit_details: list[dict[str, Any]] | None = None,
    ) -> str:
        selected_hits = [item for item in (hit_details or []) if item.get("selected")]
        if selected_hits:
            blocks: list[str] = []
            best_distance = min(float(item.get("distance") or 999999.0) for item in selected_hits)
            for idx, item in enumerate(selected_hits, start=1):
                metadata = item.get("metadata") or {}
                page_label = "—"
                page_start = metadata.get("page_start")
                page_end = metadata.get("page_end")
                if page_start is not None and page_end not in {None, page_start}:
                    page_label = f"{page_start}-{page_end}"
                elif page_start is not None:
                    page_label = str(page_start)
                best_tag = "BEST_MATCH" if float(item.get("distance") or 999999.0) == best_distance else "SUPPORTING_MATCH"
                matched_terms = metadata.get("keywords") or metadata.get("aliases") or metadata.get("question") or "—"
                header = [
                    f"SOURCE {idx} [{best_tag}]",
                    f"Раздел: {item.get('section') or metadata.get('section') or 'Без раздела'}",
                    f"Подраздел: {item.get('subsection') or metadata.get('subsection') or '—'}",
                    f"Заголовок: {item.get('title') or metadata.get('title') or '—'}",
                    f"Страницы: {page_label}",
                    f"Сигналы совпадения: {matched_terms}",
                    "Текст:",
                    item.get("text") or "",
                ]
                blocks.append("\n".join(header))
            return "\n\n-----\n\n".join(blocks)
        if sources:
            parts = [part for part in (context or "").split("\n\n---\n\n") if part.strip()]
            blocks: list[str] = []
            for idx, source in enumerate(sources[: len(parts)], start=1):
                page_start = source.get("page_start")
                page_end = source.get("page_end")
                if page_start is not None and page_end not in {None, page_start}:
                    page_label = f"{page_start}-{page_end}"
                elif page_start is not None:
                    page_label = str(page_start)
                else:
                    page_label = "—"
                best_tag = "BEST_MATCH" if idx == 1 else "SUPPORTING_MATCH"
                blocks.append(
                    "\n".join(
                        [
                            f"SOURCE {idx} [{best_tag}]",
                            f"Раздел: {source.get('section') or 'Без раздела'}",
                            f"Подраздел: {source.get('subsection') or '—'}",
                            f"Заголовок: {source.get('title') or '—'}",
                            f"Страницы: {page_label}",
                            f"Сигналы совпадения: {source.get('keywords') or source.get('aliases') or source.get('question') or '—'}",
                            "Текст:",
                            parts[idx - 1],
                        ]
                    )
                )
            if blocks:
                return "\n\n-----\n\n".join(blocks)
        return context

    def _build_user_prompt(self, question: str, structured_context: str, question_type: str, rescue: bool = False) -> str:
        if question_type == "factoid":
            instruction = (
                "Найди в ЛУЧШЕМ источнике точный факт и ответь одной короткой фразой или 1–2 предложениями. "
                "Если есть URL, адрес, число, ОС, браузер, кнопка, название раздела или термин — перенеси его в ответ без размывания. "
                "Не заменяй точный факт общим пересказом и не используй нерелевантные supporting-фрагменты, если best-match уже содержит ответ."
            )
        elif question_type == "definition":
            instruction = (
                "Дай точное определение или назначение сущности на основе лучшего источника. "
                "Если нужно, добавь одно короткое пояснение из контекста."
            )
        elif question_type == "procedure":
            instruction = (
                "Если в контексте есть действия или шаги, дай ответ строго в виде нумерованного списка шагов. "
                "Сохраняй порядок действий и точные названия кнопок/разделов. "
                "Не пропускай шаги из одного и того же раздела, даже если они разнесены по нескольким supporting-фрагментам. "
                "Если в контексте описан только переход в раздел, так и скажи и не додумывай последующие действия."
            )
        else:
            instruction = (
                "Ответь кратко, по существу, на русском языке. Используй только факты из контекста. "
                "Если ответ собирается из нескольких фрагментов, объедини их в один ответ. "
                "Если вопрос про шаги или действия — дай пошаговый ответ и не пропускай важные шаги из одного раздела. "
                "Если в контексте есть список, требования, шаги или ограничения, включи ключевые пункты в ответ, а не только общий вывод. "
                "Если в контексте есть ссылка, адрес, ограничение, срок, роль, браузер, ОС, числовое требование — обязательно включи это в ответ, когда это относится к вопросу."
            )
        if rescue:
            instruction += " Не отвечай отказом, если в контексте есть хотя бы частично полезные сведения."
        return (
            f"{instruction}\n\n"
            f"Вопрос пользователя:\n{question}\n\n"
            f"Контекст:\n{structured_context}\n\n"
            "Итоговый ответ:"
        )


    def _is_procedural_question(self, question: str) -> bool:
        normalized = (question or "").strip().lower()
        return self._classify_question(normalized) == "procedure"

    def _apply_procedure_structure(self, answer: str | None, question: str) -> str | None:
        if not answer or not self._is_procedural_question(question):
            return answer
        normalized = answer.strip()
        if not normalized:
            return answer
        if re.search(r"(?:^|\n)\s*1[\).]\s+", normalized):
            return normalized
        if ";" in normalized and normalized.count(";") >= 1:
            parts = [part.strip(" •-\n\t") for part in normalized.split(";") if part.strip()]
            if len(parts) >= 2:
                return "\n".join(f"{idx}. {part}" for idx, part in enumerate(parts, start=1))
        if normalized.count(". ") >= 2 and len(normalized) > 120:
            parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
            if 2 <= len(parts) <= 6:
                return "\n".join(f"{idx}. {part}" for idx, part in enumerate(parts, start=1))
        return normalized

    def _build_action_guard_answer(self, question: str, structured_context: str) -> str | None:
        q = (question or "").lower()
        context = (structured_context or "").lower()
        if not q or not context:
            return None

        archive_transition_only = (
            "архив" in q
            and any(token in q for token in ["архивир", "достать", "вернуть", "убрать", "восстанов", "добавить"])
            and "перейти" in context
            and "архив" in context
            and not any(token in context for token in ["архивировать", "перемест", "вернуть", "восстанов", "из архива", "убрать из архива", "добавить в архив"])
        )
        if archive_transition_only:
            return (
                "В найденном контексте описан только переход в режим архива сигналов через соответствующую кнопку или раздел. "
                "Действия по архивированию, возврату или восстановлению сигнала в этом контексте не описаны."
            )

        archive_exclamation = (
            ("восклицан" in q or "!" in q)
            and any(token in q for token in ["убрать", "снять", "удалить"])
            and "архив" in context
            and not any(token in context for token in ["убрать знак", "снять знак", "удалить знак", "квитир", "подтверд", "сброс"])
        )
        if archive_exclamation:
            return (
                "В найденном контексте описан только переход в режим архива сигналов. "
                "Прямых действий по снятию знака восклицания там нет."
            )

        menu_fixation_only = (
            any(token in q for token in ["зафиксир", "не сворачив", "было всегда", "постоянно", "закреп"])
            and "меню" in q
            and any(token in context for token in ["развернуть", "свернуть", "панель главного меню"])
            and not any(token in context for token in ["зафикс", "закреп", "не сворач", "постоянн"])
        )
        if menu_fixation_only:
            return (
                "В найденном контексте описано только разовое открытие или сворачивание бокового меню через кнопку главного меню. "
                "Постоянная фиксация или режим 'не сворачивать' в этом контексте не описаны."
            )
        return None

    def _is_refusal(self, answer: str | None) -> bool:
        normalized = (answer or "").strip().lower()
        return any(marker in normalized for marker in self._REFUSAL_MARKERS)

    def _extract_exact_fact(self, question: str, structured_context: str, *, best_only: bool = False) -> str | None:
        q = (question or "").lower()
        context = structured_context or ""
        if not context.strip():
            return None
        if best_only and "-----" in context:
            context = context.split("-----", 1)[0]

        urls = self._URL_RE.findall(context)
        if any(token in q for token in ["ссылка", "адрес", "url", "сайт"]) and urls:
            return f"Адрес системы: {urls[0]}"

        if any(token in q for token in ["оператив", "озу", "ram", "памят"]):
            match = self._MEMORY_RE.search(context) or self._GB_FALLBACK_RE.search(context)
            if match:
                return f"Рекомендуемая оперативная память: от {match.group(1)} ГБ."

        if any(token in q for token in ["браузер", "chrome", "хром"]):
            match = self._BROWSER_RE.search(context)
            if match:
                browser = match.group(0).strip().replace("  ", " ")
                return f"Рекомендуемый браузер: {browser}."
            if "браузер:" in context.lower():
                line = next((ln.strip() for ln in context.splitlines() if "браузер:" in ln.lower()), None)
                if line:
                    return line.rstrip('.').replace('● ', '').strip() + ('' if line.strip().endswith('.') else '.')

        if any(token in q for token in ["операцион", "ос", "windows", "macos", "linux", "ubuntu"]):
            os_matches = []
            for m in self._OS_RE.finditer(context):
                value = m.group(0)
                if value.lower() not in [x.lower() for x in os_matches]:
                    os_matches.append(value)
            if os_matches:
                return "Поддерживаемые операционные системы: " + ", ".join(os_matches) + "."
            if "операционная система" in context.lower():
                line = next((ln.strip() for ln in context.splitlines() if "операционная система" in ln.lower()), None)
                if line:
                    return line.rstrip('.').replace('● ', '').strip() + ('' if line.strip().endswith('.') else '.')

        if any(token in q for token in ["что такое", "кто такой", "что за", "что означает"]):
            for block in context.split("-----"):
                lines = [line.strip() for line in block.splitlines() if line.strip()]
                text_started = False
                collected = []
                for line in lines:
                    if line == "Текст:":
                        text_started = True
                        continue
                    if text_started:
                        collected.append(line)
                joined = " ".join(collected).strip()
                if joined:
                    sentence = re.split(r"(?<=[.!?])\s+", joined)[0].strip()
                    if sentence:
                        return sentence
        return None

    def _needs_contradiction_recheck(self, question: str, answer: str | None, structured_context: str) -> bool:
        if not answer or not structured_context.strip() or not self._settings.contradiction_recheck_enabled:
            return False
        question_type = self._classify_question(question)
        if question_type not in {"factoid", "definition"}:
            return False
        normalized_answer = answer.lower()
        exact_fact = self._extract_exact_fact(question, structured_context, best_only=True)
        if not exact_fact:
            return False
        if self._is_refusal(answer):
            return True
        if self._URL_RE.search(exact_fact) and not self._URL_RE.search(answer):
            return True
        if any(token in question.lower() for token in ["оператив", "памят", "озу", "ram"]) and "гб" not in normalized_answer:
            return True
        if any(token in question.lower() for token in ["браузер", "chrome", "хром"]) and "chrome" not in normalized_answer:
            return True
        if any(token in question.lower() for token in ["операцион", "ос", "windows", "macos", "linux"]) and not any(token in normalized_answer for token in ["windows", "macos", "ubuntu", "linux"]):
            return True
        return False

    def _build_recheck_prompt(self, question: str, structured_context: str, exact_fact: str) -> tuple[str, str]:
        system_prompt = self._settings.system_prompt_grounded_factoid
        user_prompt = (
            "В контексте есть точный факт, который нужно извлечь без искажений. "
            "Ответь только на вопрос пользователя, строго опираясь на контекст. "
            "Если нужный факт есть, обязательно включи его в ответ.\n\n"
            f"Вопрос пользователя:\n{question}\n\n"
            f"Подсказка по найденному факту:\n{exact_fact}\n\n"
            f"Контекст:\n{structured_context}\n\n"
            "Итоговый ответ:"
        )
        return system_prompt, user_prompt

    def generate_response(
        self,
        context: str,
        question: str,
        *,
        sources: list[dict[str, Any]] | None = None,
        hit_details: list[dict[str, Any]] | None = None,
        generation_profile: str | None = None,
    ) -> str | None:
        self._log_model("generation")
        structured_context = self._build_structured_context(context=context, sources=sources, hit_details=hit_details)
        question_type = self._classify_question(question)

        exact_fact = self._extract_exact_fact(question, structured_context, best_only=True)
        if question_type == "factoid" and exact_fact:
            logger.info("factoid extractor applied from best match")
            lowered = (question or "").lower()
            if any(token in lowered for token in ["ссылка", "адрес", "оператив", "озу", "ram", "браузер", "операцион"]):
                return exact_fact

        system_prompt = self._resolve_system_prompt(generation_profile, question_type, rescue=False)
        user_prompt = self._build_user_prompt(question=question, structured_context=structured_context, question_type=question_type, rescue=False)
        answer = self._llm_client.generate_text(system_prompt, user_prompt, temperature=0.05 if question_type == "factoid" else 0.1)

        guarded = self._build_action_guard_answer(question, structured_context)
        if guarded:
            return self._apply_procedure_structure(guarded, question)

        if question_type in {"factoid", "definition"} and (self._is_refusal(answer) or self._needs_contradiction_recheck(question, answer, structured_context)) and exact_fact:
            self._log_model("generation contradiction recheck")
            recheck_system, recheck_user = self._build_recheck_prompt(question, structured_context, exact_fact)
            rechecked = self._llm_client.generate_text(recheck_system, recheck_user, temperature=0.0)
            if rechecked:
                answer = rechecked

        if answer and not self._is_refusal(answer):
            return self._apply_procedure_structure(answer.strip(), question)
        if not self._settings.rescue_generation_enabled or not structured_context.strip():
            return answer
        self._log_model("generation rescue")
        rescue_prompt = self._resolve_system_prompt(generation_profile, question_type, rescue=True)
        rescue_user_prompt = self._build_user_prompt(question=question, structured_context=structured_context, question_type=question_type, rescue=True)
        rescued = self._llm_client.generate_text(rescue_prompt, rescue_user_prompt, temperature=0.0)
        if question_type in {"factoid", "definition"} and rescued and self._needs_contradiction_recheck(question, rescued, structured_context) and exact_fact:
            recheck_system, recheck_user = self._build_recheck_prompt(question, structured_context, exact_fact)
            second = self._llm_client.generate_text(recheck_system, recheck_user, temperature=0.0)
            return self._apply_procedure_structure(second or rescued, question)
        guarded = self._build_action_guard_answer(question, structured_context)
        if guarded:
            return self._apply_procedure_structure(guarded, question)
        return self._apply_procedure_structure(rescued or answer, question)

    def expand_query(self, question: str, max_variants: int = 3) -> list[str]:
        system_prompt = (
            f"Ты помогаешь retriever. Сгенерируй до {max_variants} кратких поисковых переформулировок. "
            "Не меняй смысл. Добавляй распространенные аббревиатуры и полные формы, если это помогает поиску. "
            "Ответь только JSON-массивом строк без пояснений."
        )
        user_prompt = f"Исходный вопрос: {question}"
        self._log_model("query expansion")
        raw = self._llm_client.generate_text(system_prompt, user_prompt, temperature=0.1)
        if not raw:
            return [question]
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            payload = json.loads(raw[start : end + 1] if start != -1 and end != -1 else raw)
            variants = [question]
            for item in payload:
                if isinstance(item, str):
                    normalized = item.strip()
                    if normalized and normalized not in variants:
                        variants.append(normalized)
                if len(variants) >= max_variants + 1:
                    break
            return variants
        except Exception:
            logger.exception("Query expansion parsing failed")
            return [question]

    def rerank_hits(self, question: str, hits: list[dict[str, Any]], top_n: int = 8) -> list[int] | None:
        if not hits:
            return None
        candidates: list[dict[str, Any]] = []
        for idx, item in enumerate(hits[:top_n], start=1):
            candidates.append(
                {
                    "id": idx,
                    "section": item.get("section"),
                    "subsection": item.get("subsection"),
                    "title": item.get("title"),
                    "record_type": item.get("record_type"),
                    "distance": item.get("distance"),
                    "text": (item.get("text") or "")[:900],
                }
            )
        system_prompt = (
            "Ты делаешь reranking для RAG. Верни JSON-массив id документов в порядке убывания полезности для ответа на вопрос. "
            "На первый план ставь фрагменты с прямым определением, пошаговой инструкцией, явным числовым требованием, ссылкой или настройкой. "
            "Для коротких фактологических вопросов приоритетны явные URL, числа, названия ОС, браузеров, кнопок, разделов и официальные определения. "
            "Не выдумывай id, используй только переданные. Без пояснений."
        )
        user_prompt = f"Вопрос: {question}\n\nКандидаты:\n{json.dumps(candidates, ensure_ascii=False)}"
        self._log_model("reranking")
        raw = self._llm_client.generate_text(system_prompt, user_prompt, temperature=0.0)
        if not raw:
            return None
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            parsed = json.loads(raw[start : end + 1] if start != -1 and end != -1 else raw)
            order = []
            allowed = {item["id"] for item in candidates}
            for item in parsed:
                if isinstance(item, int) and item in allowed and item not in order:
                    order.append(item)
            return order or None
        except Exception:
            logger.exception("Reranking parsing failed")
            return None
