"""Corpus loading and chunking for the target RAG (Retrieval-Augmented Generation) pipeline.

We sample 1k documents from NQ (Natural Questions) via BEIR (Benchmarking-IR).
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_nq_slice(n_docs: int = 1000, seed: int = 42) -> list[Document]:
    """Deterministically sample `n_docs` documents from BeIR/nq corpus."""
    ds = load_dataset("BeIR/nq", "corpus", split="corpus")

    # Sort indices so chunk order (and the index-state hash) is reproducible.
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_docs, len(ds)), replace=False)
    indices.sort()

    docs: list[Document] = []
    for i in indices.tolist():
        row = ds[i]
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        # Prepend title so the embedder gets a topical anchor.
        content = f"{title}\n\n{text}" if title else text
        if not content:
            continue
        docs.append(
            Document(
                page_content=content,
                metadata={"doc_id": str(row["_id"]), "source": "nq"},
            )
        )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split docs into ~512-char chunks; carries `doc_id` so chunks group back to source."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []
    for doc in docs:
        for j, piece in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": doc.metadata["doc_id"],
                        "source": doc.metadata.get("source", "nq"),
                        "chunk_index": j,
                    },
                )
            )
    return chunks


if __name__ == "__main__":
    # Manual smoke check: `python -m redteam.target.corpus`
    slice_ = load_nq_slice()
    print(f"Loaded {len(slice_)} NQ docs.")
    if slice_:
        print(f"doc 0 doc_id={slice_[0].metadata['doc_id']!r}")
        print(f"doc 0 first 200 chars: {slice_[0].page_content[:200]!r}")
