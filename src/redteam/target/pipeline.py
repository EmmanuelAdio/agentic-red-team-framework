"""End-to-end RAG (Retrieval-Augmented Generation) pipeline.

Composes Retriever + LLMGenerator. `run(query)` returns the full record the
executor needs to populate an exploit bundle (PROJECT_SPEC.md §7).
"""

from __future__ import annotations

from typing import Any

from redteam.config import RETRIEVER_TOP_K
from redteam.target.generator import LLMGenerator
from redteam.target.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: LLMGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, k: int = RETRIEVER_TOP_K) -> dict[str, Any]:
        """Run retrieval + generation. Returns a dict shaped for the exploit bundle."""
        retrieved = self.retriever.query(query, k=k)
        gen = self.generator.generate(query, retrieved)

        return {
            "query": query,
            # Match bundle schema fields. `is_poisoned` is set False here; the
            # executor flips it for any chunk whose doc_id is on the poison list.
            "retrieved_docs": [
                {
                    "doc_id": d.doc_id,
                    "rank": d.rank,
                    "score": d.score,
                    "content": d.content,
                    "is_poisoned": False,
                }
                for d in retrieved
            ],
            "generator_output": gen.text,
            "generator_latency_ms": gen.latency_ms,
            "prompt_template_hash": gen.prompt_template_hash,
            "index_state_hash": self.retriever.get_state_hash(),
        }
