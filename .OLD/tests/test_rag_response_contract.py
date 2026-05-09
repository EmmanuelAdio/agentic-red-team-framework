def test_rag_response_contract_and_trace_endpoint(app_client):
    ingest_response = app_client.post("/api/v1/corpus/ingest", json={})
    assert ingest_response.status_code == 200
    corpus_version = ingest_response.json()["corpus_version"]

    query_response = app_client.post(
        "/api/v1/rag/query",
        json={"query": "What does the corpus say about enterprise search?", "corpus_version": corpus_version},
    )
    assert query_response.status_code == 200

    payload = query_response.json()
    assert payload["final_answer"] is None
    assert payload["trace_id"]
    assert payload["retrieval_backend"] in {"local", "mongo"}
    assert payload["model_metadata"]["query_mode"] == "retrieve_only"
    assert isinstance(payload["retrieved_chunks"], list)
    assert len(payload["retrieved_chunks"]) > 0
    assert "Context:" in payload["prompt_used"]

    trace_response = app_client.get(f"/api/v1/traces/{payload['trace_id']}")
    assert trace_response.status_code == 200
    trace_payload = trace_response.json()["trace"]
    assert trace_payload["trace_id"] == payload["trace_id"]
    assert trace_payload["corpus_version"] == corpus_version
