from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.settings import Settings


class TextChunker:
    """Wrapper around LangChain text splitter with project defaults."""

    def __init__(self, settings: Settings):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        chunks = self._splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
