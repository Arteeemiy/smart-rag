import logging
import re
import shutil
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smart_rag.clients.vector_store_client import VectorStoreClient
from smart_rag.config import get_settings
from smart_rag.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)
BATCH_SIZE = 128


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(parts)
    return str(value).strip()


def _join_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def build_page_content(record: dict[str, Any]) -> str:
    lines: list[str] = []
    section = _clean_text(record.get("section"))
    subsection = _clean_text(record.get("subsection"))
    record_type = _clean_text(record.get("record_type"))
    title = _clean_text(record.get("title"))
    question = _clean_text(record.get("question"))
    answer = _clean_text(record.get("answer"))
    body = _clean_text(record.get("body"))
    steps = _join_list(record.get("steps"))
    key_facts = _join_list(record.get("key_facts"))
    aliases = _join_list(record.get("aliases"))
    keywords = _join_list(record.get("keywords"))
    entities = _join_list(record.get("entities"))
    ui_labels = _join_list(record.get("ui_labels"))
    preconditions = _join_list(record.get("preconditions"))
    postconditions = _join_list(record.get("postconditions"))

    if section:
        lines.append(f"Раздел: {section}")
    if subsection:
        lines.append(f"Подраздел: {subsection}")
    if record_type:
        lines.append(f"Тип записи: {record_type}")
    if title:
        lines.append(f"Заголовок: {title}")
    if question:
        lines.append(f"Вопрос: {question}")
    if answer:
        lines.append(f"Ответ: {answer}")
    if body:
        lines.append(f"Описание:\n{body}")
    if steps:
        lines.append("Шаги:\n" + "\n".join(f"- {item}" for item in steps))
    if key_facts:
        lines.append("Ключевые факты:\n" + "\n".join(f"- {item}" for item in key_facts))
    if preconditions:
        lines.append("Предусловия:\n" + "\n".join(f"- {item}" for item in preconditions))
    if postconditions:
        lines.append("Постусловия:\n" + "\n".join(f"- {item}" for item in postconditions))
    if ui_labels:
        lines.append("UI-элементы:\n" + "\n".join(f"- {item}" for item in ui_labels))
    if aliases:
        lines.append("Варианты запросов:\n" + "\n".join(f"- {item}" for item in aliases))
    if keywords:
        lines.append("Ключевые слова:\n" + ", ".join(keywords))
    if entities:
        lines.append("Сущности:\n" + ", ".join(entities))
    return "\n\n".join(item for item in lines if item).strip()


def build_metadata(record: dict[str, Any], settings) -> dict[str, Any]:
    section_path = _join_list(record.get("section_path"))
    return {
        "record_id": _clean_text(record.get("record_id")),
        "record_type": _clean_text(record.get("record_type")) or "unknown",
        "source_type": _clean_text(record.get("source_type")) or "json",
        "source_doc": _clean_text(record.get("source_doc")) or _clean_text(record.get("metadata", {}).get("source_doc")),
        "source_version": _clean_text(record.get("source_version")),
        "section": _clean_text(record.get("section")) or "unknown",
        "subsection": _clean_text(record.get("subsection")),
        "section_path": " > ".join(section_path),
        "title": _clean_text(record.get("title")),
        "question": _clean_text(record.get("question")),
        "answer_preview": _clean_text(record.get("answer"))[:500],
        "page_start": int(record.get("page_start") or 0),
        "page_end": int(record.get("page_end") or 0),
        "priority": _clean_text(record.get("priority")) or "medium",
        "chunking_hint": _clean_text(record.get("chunking_hint")) or "atomic",
        "keywords": " | ".join(_join_list(record.get("keywords"))),
        "aliases": " | ".join(_join_list(record.get("aliases"))),
        "entities": " | ".join(_join_list(record.get("entities"))),
        "ui_labels": " | ".join(_join_list(record.get("ui_labels"))),
        "index_name": settings.index_name,
        "embedding_model": settings.embedding_model,
    }


def should_split(record: dict[str, Any], text: str, settings) -> bool:
    if not text:
        return False
    hint = _clean_text(record.get("chunking_hint")).lower()
    if hint == "atomic" and len(text) <= settings.chunk_size:
        return False
    return len(text) > settings.chunk_size


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    persist_dir = Path(settings.chroma_dir)
    if settings.recreate_index and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("index=%s", settings.index_name)
    logger.info("persist_dir=%s", persist_dir)
    logger.info("collection=%s", settings.collection_name)
    logger.info("embedding=%s", settings.embedding_model)
    logger.info("data_file=%s", settings.data_path)

    repository = DocumentRepository()
    records = repository.load_json_documents(settings.data_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    vector_store_client = VectorStoreClient(settings=settings)
    vector_db = Chroma(
        collection_name=settings.collection_name,
        embedding_function=vector_store_client.embeddings,
        persist_directory=str(persist_dir),
    )

    docs: list[Document] = []
    ids: list[str] = []

    for idx, record in enumerate(records):
        page_content = build_page_content(record)
        if not page_content:
            continue
        metadata = build_metadata(record, settings)
        record_id = metadata.get("record_id") or f"record_{idx:06d}"
        chunks = splitter.split_text(page_content) if should_split(record, page_content, settings) else [page_content]
        for chunk_id, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk_id": chunk_id, "chunk_total": len(chunks)},
                )
            )
            safe_section = re.sub(r"\s+", "_", metadata.get("section") or "unknown")
            ids.append(f"{settings.index_name}:{record_id}:{safe_section}:{chunk_id}")

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        vector_db.add_documents(batch_docs, ids=batch_ids)
        logger.info("added %s/%s chunks", min(i + BATCH_SIZE, len(docs)), len(docs))

    vector_db.persist()
    logger.info("done | records=%s | chunks=%s", len(records), len(docs))


if __name__ == "__main__":
    main()
