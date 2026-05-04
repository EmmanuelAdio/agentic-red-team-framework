from pathlib import Path


def test_retrieval_returns_relevant_chunks(services):
    ingest_result = services.ingestion_service.ingest_from_path(Path("data/sample_corpus"))

    results = services.retriever.retrieve(
        query="What is retrieval augmented generation?",
        top_k=3,
        corpus_version=ingest_result.corpus_version,
    )

    assert len(results) == 3
    top_text = results[0].chunk_text.lower()
    assert "retrieval" in top_text or "generation" in top_text
    assert results[0].score >= results[-1].score
