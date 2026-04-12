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

    llm_provider: str = Field(default="mistral", alias="LLM_PROVIDER")
    llm_model: str = Field(default="mistral-small-latest", alias="LLM_MODEL")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    rag_default_mode: str = Field(default="legacy", alias="RAG_DEFAULT_MODE")
    agentic_rag_model: str = Field(default="gpt-4.1", alias="AGENTIC_RAG_MODEL")
    agentic_rag_recursion_limit: int = Field(default=8, alias="AGENTIC_RAG_RECURSION_LIMIT")

    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    mistral_api_url: str | None = Field(default=None, alias="MISTRAL_API_URL")
    mistral_small_model: str | None = Field(default="mistral-small-latest", alias="MISTRAL_SMALL_MODEL")
    mistral_medium_model: str | None = Field(default="mistral-medium-latest", alias="MISTRAL_MEDIUM_MODEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_url: str | None = Field(default=None, alias="OPENAI_API_URL")
    openai_small_model: str | None = Field(default="gpt-4.1-mini", alias="OPENAI_SMALL_MODEL")
    openai_medium_model: str | None = Field(default="gpt-4.1", alias="OPENAI_MEDIUM_MODEL")

    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_retry_delay_seconds: float = Field(default=1.0, alias="LLM_RETRY_DELAY_SECONDS")
    llm_max_retry_delay_seconds: float = Field(default=8.0, alias="LLM_MAX_RETRY_DELAY_SECONDS")
    llm_max_concurrent_requests: int = Field(default=2, alias="LLM_MAX_CONCURRENT_REQUESTS")
    llm_disable_generation_on_rate_limit: bool = Field(default=True, alias="LLM_DISABLE_GENERATION_ON_RATE_LIMIT")

    test_enable_query_expansion: bool = Field(default=True, alias="TEST_ENABLE_QUERY_EXPANSION")
    test_enable_reranking: bool = Field(default=True, alias="TEST_ENABLE_RERANKING")
    test_query_expansion_variants: int = Field(default=1, alias="TEST_QUERY_EXPANSION_VARIANTS")
    test_rerank_top_n: int = Field(default=4, alias="TEST_RERANK_TOP_N")
    test_rerank_keep_n: int | None = Field(default=None, alias="TEST_RERANK_KEEP_N")

    embedding_provider: str = Field(default="huggingface", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    index_name: str = Field(default="baseline", alias="INDEX_NAME")
    chroma_path: str = Field(default="./indexes/baseline/chroma_db", alias="CHROMA_PATH")
    collection_name: str = Field(default="knowledge_base", alias="COLLECTION_NAME")
    data_file: str = Field(default="./data/lkuk_canonical_index_ready.json", alias="DATA_FILE")
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    recreate_index: bool = Field(default=False, alias="RECREATE_INDEX")
    top_k_results: int = Field(default=8, alias="TOP_K_RESULTS")
    gate_strong: float = Field(default=7.0, alias="GATE_STRONG")
    gate_max: float = Field(default=13.0, alias="GATE_MAX")
    gap_min: float = Field(default=0.9, alias="GAP_MIN")
    window: float = Field(default=2.1, alias="WINDOW")
    min_keep: int = Field(default=2, alias="MIN_KEEP")
    max_keep: int = Field(default=5, alias="MAX_KEEP")
    export_dir: str = Field(default="./artifacts/exports", alias="EXPORT_DIR")
    google_sheets_credentials_file: str | None = Field(default=None, alias="GOOGLE_SHEETS_CREDENTIALS_FILE")

    system_prompt: str = Field(
        default=(
            "Ты — технический помощник. Отвечай строго на основе предоставленного контекста. "
            "Если ответа в контексте нет — скажи, что информации недостаточно."
        ),
        alias="SYSTEM_PROMPT",
    )
    system_prompt_grounded_v2: str = Field(
        default=(
            "Ты — технический помощник по платформе Ujin. Отвечай только по предоставленному контексту. "
            "Если в контексте есть хотя бы частично релевантные сведения, дай максимально полезный ответ на их основе. "
            "Объединяй факты из нескольких источников. Не пиши 'информации недостаточно', если из контекста можно извлечь прямой или частичный ответ. "
            "Не выдумывай факты вне контекста. Если ответа действительно нет, так и скажи коротко."
        ),
        alias="SYSTEM_PROMPT_GROUNDED_V2",
    )
    system_prompt_grounded_rescue: str = Field(
        default=(
            "Ты исправляешь слишком осторожный ответ RAG. В контексте уже есть релевантные фрагменты. "
            "Собери из них максимально полезный, конкретный и краткий ответ. "
            "Запрещено отвечать фразами вроде 'информации недостаточно', если можно назвать хотя бы определение, шаг, ссылку, ограничение, ОС, браузер, число, роль или условие. "
            "Используй только факты из контекста, без домыслов."
        ),
        alias="SYSTEM_PROMPT_GROUNDED_RESCUE",
    )
    system_prompt_grounded_factoid: str = Field(
        default=(
            "Ты отвечаешь на короткий фактологический вопрос по документации. "
            "Нужно извлечь точный факт из контекста: ссылку, адрес, термин, число, браузер, ОС, кнопку, роль, статус или ограничение. "
            "Отвечай очень коротко и точно. Если в контексте есть явный факт, процитируй его по смыслу, не заменяя общими словами. "
            "Если в лучшем источнике есть URL, число, единицы измерения, название ОС или браузера — обязательно перенеси их в ответ. "
            "Не говори, что факта нет, если он явно присутствует в контексте. Не выдумывай ничего вне контекста."
        ),
        alias="SYSTEM_PROMPT_GROUNDED_FACTOID",
    )
    system_prompt_grounded_definition: str = Field(
        default=(
            "Ты отвечаешь на вопрос-определение по документации. "
            "Дай короткое и точное определение термина, сущности или роли на основе лучшего релевантного фрагмента. "
            "Если в контексте есть официальная формулировка, используй её по смыслу. Можно добавить одно поясняющее предложение, если оно прямо следует из контекста."
        ),
        alias="SYSTEM_PROMPT_GROUNDED_DEFINITION",
    )
    system_prompt_grounded_procedure: str = Field(
        default=(
            "Ты отвечаешь на процедурный вопрос по документации. "
            "Если в контексте есть шаги или действия, верни пошаговый ответ и перечисли все существенные шаги из одного логического раздела. "
            "Не добавляй шаги, которых нет в контексте. Если шаги распределены по нескольким близким источникам внутри одного раздела, объедини их без потери точности."
        ),
        alias="SYSTEM_PROMPT_GROUNDED_PROCEDURE",
    )
    generation_profile: str = Field(default="grounded_v2", alias="GENERATION_PROFILE")
    rescue_generation_enabled: bool = Field(default=True, alias="RESCUE_GENERATION_ENABLED")
    contradiction_recheck_enabled: bool = Field(default=True, alias="CONTRADICTION_RECHECK_ENABLED")

    @property
    def chroma_dir(self) -> Path:
        return Path(self.chroma_path)

    @property
    def data_path(self) -> Path:
        return Path(self.data_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
