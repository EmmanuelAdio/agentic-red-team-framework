from __future__ import annotations

import json
from urllib import error, request
from abc import ABC, abstractmethod

from OLD.app.core.settings import Settings


class LLMGenerator(ABC):
    """Interface for answer generation on top of retrieved context."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate an answer from a prepared prompt."""


class NoopGenerator(LLMGenerator):
    """Placeholder generator for retrieve-only mode."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("No generator configured for retrieve-only mode")


class OpenAIGenerator(LLMGenerator):
    """OpenAI chat completion adapter."""

    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LLM generation requires package 'openai'. Install it with: pip install openai"
            ) from exc

        if not settings.openai_api_key:
            raise RuntimeError(
                "OpenAI provider selected but OPENAI_API_KEY is not configured in environment/.env."
            )

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model if settings.llm_model != "none" else "gpt-4o-mini"

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


class StubGenerator(OpenAIGenerator):
    """Backward-compatible alias for legacy provider name."""


class OllamaGenerator(LLMGenerator):
    """Local model adapter via Ollama HTTP API."""

    def __init__(self, settings: Settings):
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.llm_model if settings.llm_model != "none" else "llama3.1"
        self._timeout = settings.ollama_timeout_seconds

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(
                "Failed to call local Ollama server. Ensure Ollama is running and OLLAMA_BASE_URL is correct."
            ) from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid JSON response from Ollama /api/generate endpoint.") from exc

        response_text = data.get("response")
        if not isinstance(response_text, str):
            raise RuntimeError("Ollama response payload missing expected 'response' text.")
        return response_text


def build_generator(settings: Settings) -> LLMGenerator:
    """Factory for generation adapters based on configured query mode/provider."""

    if settings.query_mode == "retrieve_only" or settings.llm_provider == "none":
        return NoopGenerator()

    provider = settings.llm_provider.lower()

    if provider in {"openai", "stub"}:
        return StubGenerator(settings)
    if provider in {"ollama", "local_model", "local"}:
        return OllamaGenerator(settings)

    raise ValueError(f"Unsupported llm provider: {settings.llm_provider}")
