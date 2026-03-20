from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Agentic Red-Team API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "agentic_red_team_baseline"
    mongo_use_mock: bool = False

    default_corpus_path: Path = Field(default=Path("data/sample_corpus"))

    vector_backend: Literal["local", "mongo"] = "local"

    embedding_provider: str = "deterministic_stub"
    embedding_model: str = "hash-bow-v1"
    embedding_dimension: int = 256

    llm_provider: str = "none"
    llm_model: str = "none"
    query_mode: Literal["retrieve_only", "generate"] = "retrieve_only"

    default_top_k: int = 4
    chunk_size: int = 400
    chunk_overlap: int = 60


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for the running process."""

    return Settings()
