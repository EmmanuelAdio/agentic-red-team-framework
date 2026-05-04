from __future__ import annotations

import io
import json
import sys
import types
from types import SimpleNamespace

import pytest

from OLD.app.core.settings import Settings
from OLD.app.rag.generator import NoopGenerator, OllamaGenerator, OpenAIGenerator, build_generator


def test_build_generator_returns_noop_for_retrieve_only():
    settings = Settings(query_mode="retrieve_only", llm_provider="openai", llm_model="gpt-4o-mini")
    generator = build_generator(settings)
    assert isinstance(generator, NoopGenerator)


def test_build_generator_returns_openai_for_openai_provider(monkeypatch: pytest.MonkeyPatch):
    class FakeCompletions:
        def create(self, **_: object):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    class FakeClient:
        def __init__(self, api_key: str | None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    settings = Settings(
        query_mode="generate",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        openai_api_key="test-key",
    )
    generator = build_generator(settings)

    assert isinstance(generator, OpenAIGenerator)
    assert generator.generate("hello") == "ok"


def test_build_generator_returns_ollama_for_local_provider():
    settings = Settings(query_mode="generate", llm_provider="local_model", llm_model="llama3.1")
    generator = build_generator(settings)
    assert isinstance(generator, OllamaGenerator)


def test_ollama_generator_calls_http_endpoint(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"response": "local answer"}).encode("utf-8")

    def fake_urlopen(req, timeout: float):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["method"] = req.method
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("app.rag.generator.request.urlopen", fake_urlopen)
    settings = Settings(
        query_mode="generate",
        llm_provider="ollama",
        llm_model="llama3.1",
        ollama_base_url="http://localhost:11434",
        ollama_timeout_seconds=7,
    )
    generator = build_generator(settings)
    result = generator.generate("test prompt")

    payload = json.loads(captured["body"].decode("utf-8"))  # type: ignore[union-attr]
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 7.0
    assert captured["method"] == "POST"
    assert payload["model"] == "llama3.1"
    assert payload["prompt"] == "test prompt"
    assert result == "local answer"


def test_ollama_generator_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch):
    class FakeBadResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return io.BytesIO(b"not-json").read()

    def fake_urlopen(req, timeout: float):
        return FakeBadResponse()

    monkeypatch.setattr("app.rag.generator.request.urlopen", fake_urlopen)
    settings = Settings(query_mode="generate", llm_provider="ollama", llm_model="llama3.1")
    generator = build_generator(settings)

    with pytest.raises(RuntimeError, match="Invalid JSON response from Ollama"):
        generator.generate("prompt")
