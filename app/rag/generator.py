from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.settings import Settings


class LLMGenerator(ABC):
    """Interface for answer generation on top of retrieved context."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate an answer from a prepared prompt."""


class NoopGenerator(LLMGenerator):
    """Placeholder generator for retrieve-only mode."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("No generator configured for retrieve-only mode")


class StubGenerator(LLMGenerator):
    """Deterministic placeholder generator for future provider wiring."""

    def generate(self, prompt: str) -> str:
        # TODO(provider): replace with real model invocation.
        return "Stub answer: replace generator adapter with a real LLM provider."


def build_generator(settings: Settings) -> LLMGenerator:
    """Factory for generation adapters based on configured query mode/provider."""

    if settings.query_mode == "retrieve_only" or settings.llm_provider == "none":
        return NoopGenerator()

    if settings.llm_provider == "stub":
        return StubGenerator()

    # TODO(provider): add real LLM adapters (OpenAI, local models, etc.)
    raise ValueError(f"Unsupported llm provider: {settings.llm_provider}")
