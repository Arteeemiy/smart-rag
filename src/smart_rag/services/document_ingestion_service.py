from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..clients.vector_store_client import VectorStoreClient
from ..config import Settings


@dataclass(slots=True)
class UploadedKnowledgeBase:
    vector_store: object
    documents: list[Any]
    filename: str
    content_type: str | None
    chunk_count: int


class DocumentIngestionService:
    def __init__(self, vector_store_client: VectorStoreClient, settings: Settings) -> None:
        self._vector_store_client = vector_store_client
        self._settings = settings

    def _split_documents(self, documents: list[Any]) -> list[Any]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def _load_pdf(self, path: Path, filename: str) -> list[Any]:
        from langchain_community.document_loaders import PyPDFLoader

        pages = PyPDFLoader(str(path)).load()
        for index, page in enumerate(pages, start=1):
            page.metadata = {
                **(page.metadata or {}),
                "source_doc": filename,
                "source_type": "uploaded_document",
                "section": f"Page {index}",
                "page_start": index,
                "page_end": index,
            }
        return pages

    def _load_text_like(self, raw_text: str, filename: str) -> list[Any]:
        from langchain.schema import Document

        return [
            Document(
                page_content=raw_text,
                metadata={
                    "source_doc": filename,
                    "source_type": "uploaded_document",
                    "section": "Uploaded document",
                },
            )
        ]

    def _load_documents(self, path: Path, *, filename: str, content_type: str | None) -> list[Any]:
        suffix = path.suffix.lower()
        if content_type == "application/pdf" or suffix == ".pdf":
            return self._load_pdf(path, filename)
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        return self._load_text_like(raw_text=raw_text, filename=filename)

    def build_uploaded_knowledge_base(
        self,
        *,
        filename: str,
        file_bytes: bytes,
        content_type: str | None,
    ) -> UploadedKnowledgeBase:
        suffix = Path(filename or "uploaded.txt").suffix or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(file_bytes)
            temp_path = Path(handle.name)
        try:
            documents = self._load_documents(temp_path, filename=filename, content_type=content_type)
        finally:
            temp_path.unlink(missing_ok=True)

        split_documents = self._split_documents(documents)
        for index, doc in enumerate(split_documents, start=1):
            doc.metadata = {
                **(doc.metadata or {}),
                "record_id": f"upload-{index}",
                "record_type": "uploaded_chunk",
                "title": filename,
                "chunk_index": index,
            }
        vector_store = self._vector_store_client.build_in_memory_vector_store(split_documents)
        return UploadedKnowledgeBase(
            vector_store=vector_store,
            documents=split_documents,
            filename=filename,
            content_type=content_type,
            chunk_count=len(split_documents),
        )
