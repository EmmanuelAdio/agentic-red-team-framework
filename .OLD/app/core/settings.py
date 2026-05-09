from pathlib import Path
from threading import RLock
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
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 30.0
    atlas_vector_index_name: str = "chunks_embedding_index"

    default_top_k: int = 4
    chunk_size: int = 400
    chunk_overlap: int = 60


def get_settings() -> Settings:
    """Return settings and refresh automatically when `.env` changes."""

    global _cached_settings, _cached_settings_key

    settings_key = get_settings_cache_key()
    with _settings_lock:
        if _cached_settings is None or _cached_settings_key != settings_key:
            _cached_settings = Settings()
            _cached_settings_key = settings_key
        return _cached_settings


def clear_settings_cache() -> None:
    """Reset cached settings, forcing a reload on next access."""

    global _cached_settings, _cached_settings_key

    with _settings_lock:
        _cached_settings = None
        _cached_settings_key = None


def get_settings_cache_key() -> int:
    """Fingerprint for config reload decisions based on `.env` mtime."""

    env_path = Path(".env")
    if not env_path.exists():
        return -1
    return env_path.stat().st_mtime_ns


_settings_lock = RLock()
_cached_settings: Settings | None = None
_cached_settings_key: int | None = None

# Backwards-compatible test API: existing tests call `get_settings.cache_clear()`.
get_settings.cache_clear = clear_settings_cache  # type: ignore[attr-defined]
