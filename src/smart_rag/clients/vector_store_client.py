from __future__ import annotations

from typing import Any

from ..config import Settings
from .embeddings_factory import create_embeddings


class VectorStoreClient:
    def __init__(self, settings: Settings) -> None:
        from langchain_community.vectorstores import Chroma

        self._settings = settings
        self._embeddings = create_embeddings(settings)
        self._vector_store = Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.chroma_dir),
            embedding_function=self._embeddings,
        )

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def embeddings(self):
        return self._embeddings

    def similarity_search_with_score(self, query: str, *, k: int, vector_store: Any | None = None):
        store = vector_store or self._vector_store
        return store.similarity_search_with_score(query, k=k)

    def build_in_memory_vector_store(self, documents: list[Any]):
        from langchain_core.vectorstores import InMemoryVectorStore

        vector_store = InMemoryVectorStore(self._embeddings)
        vector_store.add_documents(documents)
        return vector_store
