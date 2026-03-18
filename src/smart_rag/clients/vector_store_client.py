from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from ..config import Settings


class VectorStoreClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self._vector_store = Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.chroma_dir),
            embedding_function=self._embeddings,
        )

    @property
    def vector_store(self) -> Chroma:
        return self._vector_store

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings
