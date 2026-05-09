from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure `app` imports resolve regardless of where pytest is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OLD.app.core.container import get_container
from OLD.app.core.settings import get_settings
from OLD.app.main import create_app


@pytest.fixture(autouse=True)
def isolated_settings_env(monkeypatch: pytest.MonkeyPatch):
    """Use isolated in-memory Mongo per test run and reset cached singletons."""

    monkeypatch.setenv("MONGO_USE_MOCK", "true")
    monkeypatch.setenv("MONGODB_DB_NAME", f"test_db_{uuid.uuid4().hex[:8]}")
    monkeypatch.setenv("DEFAULT_CORPUS_PATH", "data/sample_corpus")
    monkeypatch.setenv("VECTOR_BACKEND", os.getenv("VECTOR_BACKEND", "local"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "deterministic_stub")
    monkeypatch.setenv("EMBEDDING_MODEL", "hash-bow-v1")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "256")
    monkeypatch.setenv("QUERY_MODE", "retrieve_only")
    monkeypatch.setenv("LLM_PROVIDER", "none")

    get_settings.cache_clear()
    get_container.cache_clear()
    yield
    get_container.cache_clear()
    get_settings.cache_clear()


@pytest.fixture()
def app_client() -> TestClient:
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def services():
    return get_container()
