from decimal import Decimal
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Service"
    app_version: str = "0.1.0"
    app_env: str = "development"
    cors_origins: list[str] = ["*"]

    rag_api_key: str = "change-this-secret"
    database_url: str

    ai_provider: str = "openai"

    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_chat_model: str = "gemini-2.5-flash-lite"

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-5.4-mini"
    openai_ocr_model: str = "gpt-4o-mini"

    openai_embedding_price_per_million_tokens: Decimal = Decimal("0.02")
    openai_chat_input_price_per_million_tokens: Decimal = Decimal("0.75")
    openai_chat_cached_input_price_per_million_tokens: Decimal = Decimal("0.075")
    openai_chat_output_price_per_million_tokens: Decimal = Decimal("4.50")
    openai_ocr_input_price_per_million_tokens: Decimal = Decimal("0.15")
    openai_ocr_cached_input_price_per_million_tokens: Decimal = Decimal("0.075")
    openai_ocr_output_price_per_million_tokens: Decimal = Decimal("0.60")

    upload_dir: str = "/code/app/uploads"
    top_k: int = 5
    embedding_dim: int = 768

    # Legacy character settings remain available for compatibility.
    max_chunk_chars: int = 7000
    chunk_overlap_chars: int = 500

    # Preferred token-aware chunking settings.
    max_chunk_tokens: int = 900
    chunk_overlap_tokens: int = 120
    max_chunks_per_document: int = 0
    embedding_request_delay_seconds: float = 0.0

    # Hybrid retrieval and local reranking.
    retrieval_candidate_k: int = 20
    min_relevance_score: float = 0.20
    vector_weight: float = 0.60
    lexical_weight: float = 0.25
    exact_match_weight: float = 0.15
    neighbor_window: int = 1
    max_context_chunks: int = 12

    enable_ocr_fallback: bool = False
    max_ocr_pages: int = 10
    ocr_render_zoom: float = 2.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
