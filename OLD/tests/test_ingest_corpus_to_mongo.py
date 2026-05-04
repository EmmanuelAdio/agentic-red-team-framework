import mongomock

from OLD.scripts.ingest_corpus_to_mongo import (
    DeterministicStubEmbeddingBackend,
    ensure_indexes,
    ingest_rows,
)


def _row(doc_id: str, doc_type: str, attack_type=None):
    return {
        "doc_id": doc_id,
        "entity_id": f"entity::{doc_id}",
        "entity_type": "wiki",
        "title": "Title",
        "source": "source",
        "source_type": "wiki",
        "content": "This is a small test document.",
        "doc_type": doc_type,
        "attack_type": attack_type,
        "tags": ["x"],
        "metadata": {},
    }


def test_ingest_corpus_to_mongo_inserts_embeddings_indexes_and_clear_first():
    client = mongomock.MongoClient()
    collection = client["test_db"]["rag_documents"]

    rows = [_row("d1", "benign"), _row("d2", "poisoned", "prompt_injection")]
    backend = DeterministicStubEmbeddingBackend(dimension=32)

    stats = ingest_rows(collection, rows, backend, clear_first=True, batch_size=1)
    assert stats["inserted_count"] == 2
    assert collection.count_documents({}) == 2

    stored = collection.find_one({"doc_id": "d1"})
    assert stored is not None
    assert isinstance(stored.get("embedding"), list)
    assert len(stored["embedding"]) == 32

    ensure_indexes(collection)
    index_names = {index["name"] for index in collection.list_indexes()}
    assert "doc_id_1" in index_names
    assert "entity_id_1" in index_names
    assert "entity_type_1" in index_names
    assert "doc_type_1" in index_names
    assert "attack_type_1" in index_names
    assert "source_type_1" in index_names

    stats_second = ingest_rows(collection, [_row("d3", "benign")], backend, clear_first=True, batch_size=10)
    assert stats_second["inserted_count"] == 1
    assert collection.count_documents({}) == 1
