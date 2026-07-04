from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Service"
    app_version: str = "0.1.0"
    app_env: str = "development"
    cors_origins: list[str] = ["*"]

    rag_api_key: str = "change-this-secret"
    database_url: str

    gemini_api_key: str
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_chat_model: str = "gemini-2.5-flash-lite"

    upload_dir: str = "/code/app/uploads"
    top_k: int = 5
    embedding_dim: int = 768
    max_chunk_chars: int = 7000
    chunk_overlap_chars: int = 500
    max_chunks_per_document: int = 20
    embedding_request_delay_seconds: float = 1.5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
