from __future__ import annotations

from OLD.app.rag.schemas import RetrievedChunk


def build_prompt_context(chunks: list[RetrievedChunk]) -> str:
    """Assemble retrieved chunk context for prompt construction."""

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"[{index}] ({chunk.chunk_id}) {chunk.chunk_text}")
    return "\n".join(lines)


def build_prompt(query: str, prompt_context: str) -> str:
    """Build baseline prompt template with retrieval context."""

    return (
        "You are a baseline question-answering assistant. "
        "Answer only from the provided context. "
        "If context is insufficient, say that clearly.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{prompt_context}\n\n"
        "Answer:"
    )
