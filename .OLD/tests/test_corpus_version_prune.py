from OLD.app.core.container import get_container


def test_prune_corpus_versions_deletes_old_versions_and_related_records(app_client):
    app_client.post("/api/v1/corpus/ingest", json={"source_path": "data/sample_corpus"})
    app_client.post("/api/v1/corpus/ingest", json={"source_path": "data/sample_corpus"})
    latest_response = app_client.post("/api/v1/corpus/ingest", json={"source_path": "data/sample_corpus"})
    assert latest_response.status_code == 200
    latest_version = latest_response.json()["corpus_version"]

    services = get_container()
    before_version_count = services.repositories.corpus_version_repo.collection.count_documents({})
    before_doc_count = services.repositories.document_repo.collection.count_documents({})
    before_chunk_count = services.repositories.chunk_repo.collection.count_documents({})
    assert before_version_count == 3
    assert before_doc_count > 0
    assert before_chunk_count > 0

    prune_response = app_client.post("/api/v1/corpus/versions/prune", json={"delete_count": 2, "keep_latest": True})
    assert prune_response.status_code == 200
    payload = prune_response.json()

    assert payload["deleted_version_count"] == 2
    assert payload["deleted_document_count"] > 0
    assert payload["deleted_chunk_count"] > 0
    assert payload["deleted_corpus_version_count"] == 2
    assert len(payload["deleted_versions"]) == 2

    after_version_count = services.repositories.corpus_version_repo.collection.count_documents({})
    after_doc_count = services.repositories.document_repo.collection.count_documents({})
    after_chunk_count = services.repositories.chunk_repo.collection.count_documents({})
    assert after_version_count == 1
    assert after_doc_count < before_doc_count
    assert after_chunk_count < before_chunk_count

    latest_row = services.repositories.corpus_version_repo.get(latest_version)
    assert latest_row is not None


def test_prune_corpus_versions_returns_zero_when_no_versions(app_client):
    response = app_client.post("/api/v1/corpus/versions/prune", json={"delete_count": 1, "keep_latest": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted_version_count"] == 0
    assert payload["deleted_versions"] == []


def test_prune_corpus_versions_also_cleans_orphan_documents_and_chunks(app_client):
    ingest_response = app_client.post("/api/v1/corpus/ingest", json={"source_path": "data/sample_corpus"})
    assert ingest_response.status_code == 200
    corpus_version = ingest_response.json()["corpus_version"]

    services = get_container()
    assert services.repositories.document_repo.collection.count_documents({"corpus_version": corpus_version}) > 0
    assert services.repositories.chunk_repo.collection.count_documents({"corpus_version": corpus_version}) > 0

    # Simulate a manual/incomplete delete where only corpus_versions rows were removed.
    services.repositories.corpus_version_repo.collection.delete_one({"corpus_version": corpus_version})
    assert services.repositories.corpus_version_repo.get(corpus_version) is None

    prune_response = app_client.post("/api/v1/corpus/versions/prune", json={"delete_count": 1, "keep_latest": True})
    assert prune_response.status_code == 200
    payload = prune_response.json()

    assert payload["deleted_version_count"] == 0
    assert payload["deleted_document_count"] > 0
    assert payload["deleted_chunk_count"] > 0
    assert services.repositories.document_repo.collection.count_documents({"corpus_version": corpus_version}) == 0
    assert services.repositories.chunk_repo.collection.count_documents({"corpus_version": corpus_version}) == 0
