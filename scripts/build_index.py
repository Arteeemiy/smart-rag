import logging
import re
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smart_rag.clients.vector_store_client import VectorStoreClient
from smart_rag.config import get_settings
from smart_rag.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)
BATCH_SIZE = 128


def strip_section_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("раздел:"):
        return "\n".join(lines[1:]).lstrip()
    return text


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

    repository = DocumentRepository()
    data = repository.load_json_documents(settings.data_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    vector_store_client = VectorStoreClient(settings=settings)
    vector_db = Chroma(
        collection_name=settings.collection_name,
        embedding_function=vector_store_client.embeddings,
        persist_directory=str(persist_dir),
    )

    docs: list[Document] = []
    ids: list[str] = []

    for idx, item in enumerate(data):
        raw_text = (item.get("content") or "").strip()
        if not raw_text:
            continue
        metadata = item.get("metadata") or {}
        section = str(metadata.get("section") or "unknown")
        text = strip_section_header(raw_text)
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for chunk_id, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "index_name": settings.index_name,
                        "chunk_size": settings.chunk_size,
                        "chunk_overlap": settings.chunk_overlap,
                        "embedding_model": settings.embedding_model,
                        "chunk_id": chunk_id,
                    },
                )
            )
            safe_section = re.sub(r"\s+", "_", section.strip())
            ids.append(f"{settings.index_name}:{idx}:{safe_section}:{chunk_id}")

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        vector_db.add_documents(batch_docs, ids=batch_ids)
        logger.info("added %s/%s chunks", min(i + BATCH_SIZE, len(docs)), len(docs))

    vector_db.persist()
    logger.info("done")


if __name__ == "__main__":
    main()
