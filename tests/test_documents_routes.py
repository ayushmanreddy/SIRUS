from __future__ import annotations

import fitz
import pytest

from app.rag.prompts import NO_EVIDENCE_ANSWER


def _upload(client, filename: str, data: bytes, content_type: str = "text/plain"):
    return client.post(
        "/documents/ingest",
        files={"file": (filename, data, content_type)},
    )


def test_ingest_and_duplicate_upload(client):
    response = _upload(client, "handbook.txt", b"Paris is the capital of France.")
    assert response.status_code == 200
    payload = response.json()
    assert payload["duplicate"] is False
    assert payload["chunk_count"] >= 1
    assert payload["chunks"][0]["chunk_id"]

    duplicate = _upload(client, "renamed.txt", b"Paris is the capital of France.")
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document_id"] == payload["document_id"]


def test_ingest_rejects_oversize_upload(client, container):
    container.settings.max_upload_mb = 1
    response = _upload(client, "large.txt", b"x" * (1024 * 1024 + 1))
    assert response.status_code == 413


def test_ingest_rejects_unsupported_type(client):
    response = _upload(client, "image.png", b"not an image", "image/png")
    assert response.status_code == 415


def test_ingest_rejects_empty_file(client):
    response = _upload(client, "empty.txt", b"")
    assert response.status_code == 422


def test_ingest_rejects_corrupted_pdf(client):
    response = _upload(client, "broken.pdf", b"not a pdf", "application/pdf")
    assert response.status_code == 422


def test_ingest_native_text_pdf(client):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "SIRUS PDF pipeline extracts this native text correctly.", fontsize=14)
    data = document.write()
    document.close()
    response = _upload(client, "report.pdf", data, "application/pdf")
    assert response.status_code == 200
    assert response.json()["page_count"] == 1


def test_upload_filename_is_sanitized(client):
    response = _upload(client, "../../evil.txt", b"Safe content for traversal test.")
    assert response.status_code == 200
    assert response.json()["filename"] == "evil.txt"


def test_list_and_get_document(client):
    created = _upload(client, "policy.txt", b"Policy document content for listing.").json()
    listing = client.get("/documents")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["documents"][0]["document_id"] == created["document_id"]

    detail = client.get(f"/documents/{created['document_id']}")
    assert detail.status_code == 200
    assert detail.json()["filename"] == "policy.txt"
    assert detail.json()["chunks"]


def test_get_missing_document_returns_404(client):
    response = client.get("/documents/unknown")
    assert response.status_code == 404


def test_query_returns_answer_and_traceable_sources(client):
    created = _upload(
        client,
        "france.txt",
        b"Paris is the capital of France. The Eiffel Tower is in Paris.",
    ).json()
    response = client.post(
        "/documents/query",
        json={"question": "What is the capital of France?", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["grounded"] is True
    assert payload["sources"]
    source = payload["sources"][0]
    assert source["document_id"] == created["document_id"]
    assert source["document_name"] == "france.txt"
    assert source["chunk_id"]
    assert source["page"] == 1
    assert source["citation_id"] == 1
    assert "text_preview" in source
    assert source["score"] >= 0


def test_query_before_upload_returns_no_evidence(client):
    response = client.post("/documents/query", json={"question": "What is the policy?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == NO_EVIDENCE_ANSWER
    assert payload["sources"] == []
    assert payload["grounded"] is False


def test_query_rejects_blank_question(client):
    response = client.post("/documents/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_returns_503_when_llm_is_unavailable(client, container):
    _upload(client, "france.txt", b"Paris is the capital of France.")
    container.llm_client.unavailable = True
    response = client.post("/documents/query", json={"question": "capital of France"})
    assert response.status_code == 503


def test_query_returns_503_when_embeddings_unavailable(client, container):
    _upload(client, "france.txt", b"Paris is the capital of France.")
    container.embedding_provider.unavailable = True
    response = client.post("/documents/query", json={"question": "capital of France"})
    assert response.status_code == 503


def test_query_returns_503_when_vector_store_unavailable(client, container):
    _upload(client, "france.txt", b"Paris is the capital of France.")
    container.vector_store.unavailable = True
    response = client.post("/documents/query", json={"question": "capital of France"})
    assert response.status_code == 503


def test_query_unknown_document_filter_returns_404(client):
    _upload(client, "france.txt", b"Paris is the capital of France.")
    response = client.post(
        "/documents/query",
        json={"question": "capital", "document_id": "does-not-exist"},
    )
    assert response.status_code == 404


def test_analyze_document(client):
    response = client.post(
        "/documents/analyze",
        files={"file": ("report.txt", b"Report text", "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["extracted_text"] == "Report text"
    assert payload["summary"]
    assert payload["page_count"] == 1


def test_health_live_and_ready(client):
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "live"}

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_health_ready_degraded_when_dependency_unavailable(client, container):
    container.vector_store.unavailable = True
    ready = client.get("/health/ready")
    assert ready.status_code == 503
    payload = ready.json()
    assert payload["status"] == "degraded"
    checks = {check["name"]: check["status"] for check in payload["checks"]}
    assert checks["vector_store"] == "unavailable"


def test_system_endpoints(client):
    version = client.get("/system/version")
    assert version.status_code == 200
    assert version.json()["name"]

    status = client.get("/system/status")
    assert status.status_code == 200
    body = status.json()
    assert "models" in body
    assert "checks" in body

    preflight = client.get("/system/preflight")
    assert preflight.status_code == 200
    assert "hardware" in preflight.json()
