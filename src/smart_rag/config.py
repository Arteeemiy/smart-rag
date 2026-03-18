from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = Field(default="smart-rag", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")

    llm_api_url: str | None = Field(default=None, alias="LLM_API_URL")
    llm_model: str = Field(default="your-llm-model-name", alias="LLM_MODEL")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    llm_provider: str = Field(default="mistral", alias="LLM_PROVIDER")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    index_name: str = Field(default="baseline", alias="INDEX_NAME")
    chroma_path: str = Field(
        default="./indexes/baseline/chroma_db", alias="CHROMA_PATH"
    )
    collection_name: str = Field(default="knowledge_base", alias="COLLECTION_NAME")
    data_file: str = Field(
        default="./data/sample_documents/sample_documents.json", alias="DATA_FILE"
    )
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    recreate_index: bool = Field(default=False, alias="RECREATE_INDEX")
    top_k_results: int = Field(default=8, alias="TOP_K_RESULTS")
    gate_strong: float = Field(default=7.0, alias="GATE_STRONG")
    gate_max: float = Field(default=13.0, alias="GATE_MAX")
    gap_min: float = Field(default=0.9, alias="GAP_MIN")
    window: float = Field(default=2.1, alias="WINDOW")
    min_keep: int = Field(default=2, alias="MIN_KEEP")
    max_keep: int = Field(default=5, alias="MAX_KEEP")
    system_prompt: str = Field(
        default=(
            "Ты — технический помощник. Отвечай строго на основе предоставленного контекста. "
            "Если ответа в контексте нет — скажи, что информации недостаточно."
        ),
        alias="SYSTEM_PROMPT",
    )

    @property
    def chroma_dir(self) -> Path:
        return Path(self.chroma_path)

    @property
    def data_path(self) -> Path:
        return Path(self.data_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
