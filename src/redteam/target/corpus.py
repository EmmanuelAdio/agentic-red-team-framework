"""Corpus loading and chunking for the target RAG (Retrieval-Augmented Generation) pipeline.

Stratified 1k-doc sample of NQ (Natural Questions) from BEIR (Benchmarking-IR):
gold docs for `n_queries` test queries are guaranteed in the slice; the remainder
is filled by uniform random sampling. This is what makes Day 9's experiment matrix
(50 queries answerable from corpus) feasible.
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face dataset names (BeIR repackages NQ for the BEIR benchmark suite).
HF_NQ_CORPUS = "BeIR/nq"
HF_NQ_QRELS = "BeIR/nq-qrels"


def select_test_queries(
    n_queries: int = 50,
    seed: int = 42,
    qrels_split: str = "test",
) -> list[tuple[str, str, list[str]]]:
    """Pick `n_queries` NQ test queries with their gold doc_ids — deterministic.

    Returns list of (query_id, query_text, gold_doc_ids).
    Single source of truth so that the corpus loader and queries.json builder agree.
    """
    qrels = load_dataset(HF_NQ_QRELS, split=qrels_split)
    queries = load_dataset(HF_NQ_CORPUS, "queries", split="queries")

    qid_to_text: dict[str, str] = {row["_id"]: row["text"] for row in queries}

    # Group qrels by query_id, keeping only positive judgements (score >= 1).
    qid_to_gold: dict[str, list[str]] = {}
    for row in qrels:
        if row["score"] >= 1:
            qid_to_gold.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))

    # Keep only queries whose text is available; sort for deterministic order.
    candidates = sorted(
        (qid, qid_to_text[qid], golds)
        for qid, golds in qid_to_gold.items()
        if qid in qid_to_text
    )

    rng = np.random.default_rng(seed)
    take = min(n_queries, len(candidates))
    indices = rng.choice(len(candidates), size=take, replace=False)
    indices.sort()
    return [candidates[i] for i in indices.tolist()]


def load_nq_slice(
    n_docs: int = 1000,
    n_queries: int = 50,
    seed: int = 42,
) -> list[Document]:
    """Build a stratified 1k-doc NQ slice.

    Stage 1: include every gold doc for the `n_queries` test queries.
    Stage 2: top up with uniform random docs (excluding gold) to reach `n_docs`.

    Returns LangChain Documents with metadata `doc_id`, `source`, `is_gold`.
    """
    selected = select_test_queries(n_queries=n_queries, seed=seed)
    gold_ids: set[str] = {gid for _, _, golds in selected for gid in golds}

    ds = load_dataset(HF_NQ_CORPUS, "corpus", split="corpus")

    # Materialise the id column once; gives O(1) id->index lookups thereafter.
    # ~2.6M strings ≈ 150 MB — acceptable for a one-shot build script.
    id_column = ds["_id"]
    id_to_idx: dict[str, int] = {str(_id): i for i, _id in enumerate(id_column)}

    # Resolve gold doc indices (skip any gold id missing from the corpus split).
    gold_indices = sorted({id_to_idx[gid] for gid in gold_ids if gid in id_to_idx})

    # Random fill: oversample so we can skip any indices that overlap gold.
    n_random = max(0, n_docs - len(gold_indices))
    rng = np.random.default_rng(seed)
    over = rng.choice(
        len(ds),
        size=min(len(ds), n_random + len(gold_indices) + 100),
        replace=False,
    )
    gold_idx_set = set(gold_indices)
    fill_indices: list[int] = []
    for idx in over.tolist():
        if idx in gold_idx_set:
            continue
        fill_indices.append(int(idx))
        if len(fill_indices) >= n_random:
            break

    # Sort union for reproducible chunk ordering (drives index_state_hash).
    all_indices = sorted(set(gold_indices) | set(fill_indices))

    docs: list[Document] = []
    for i in all_indices:
        row = ds[i]
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        content = f"{title}\n\n{text}" if title else text
        if not content:
            continue
        rid = str(row["_id"])
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "doc_id": rid,
                    "source": "nq",
                    "is_gold": rid in gold_ids,
                },
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
                        "is_gold": doc.metadata.get("is_gold", False),
                        "chunk_index": j,
                    },
                )
            )
    return chunks


if __name__ == "__main__":
    # Manual smoke check: `python -m redteam.target.corpus`
    queries = select_test_queries()
    print(f"Selected {len(queries)} test queries.")
    print(f"  e.g. q0 = {queries[0][0]} -> {queries[0][1]!r}")

    slice_ = load_nq_slice()
    n_gold = sum(1 for d in slice_ if d.metadata["is_gold"])
    print(f"Loaded {len(slice_)} NQ docs ({n_gold} gold, {len(slice_) - n_gold} random fill).")
    if slice_:
        print(f"doc 0 doc_id={slice_[0].metadata['doc_id']!r} is_gold={slice_[0].metadata['is_gold']}")
