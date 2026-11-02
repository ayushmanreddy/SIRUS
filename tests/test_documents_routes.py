import pytest
from fastapi.testclient import TestClient

import app.main as app_module
import app.ocr as ocr_mod
from app.ai._vector_store import VectorStoreMetadata, VectorStoreSearchResult
from app.ai.vector_store_qdrant import QdrantClient
from app.config import Settings
from app.ocr import InvalidPDFError

client = TestClient(app_module.app)


def test_documents_ingest_oversize_upload():
    """/documents/ingest rejects a file larger than max_upload_mb with 413."""
    max_size = Settings().max_upload_mb * 1024 * 1024
    large_bytes = b"x" * (max_size + 1)
    response = client.post(
        "/documents/ingest",
        files={"file": ("large.pdf", large_bytes, "application/pdf")},
    )
    assert response.status_code == 413


def test_documents_ingest_empty_ocr_text_422(monkeypatch):
    """/documents/ingest returns 422 when OCR produces no text."""
    import fitz
    from unittest.mock import MagicMock

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test", fontsize=18)
    pdf_bytes = doc.write()
    doc.close()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []
    monkeypatch.setattr(ocr_mod, "_get_reader", lambda: mock_reader)

    response = client.post(
        "/documents/ingest",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 422


def _patch_query_dependencies(monkeypatch, search_results, answer):
    async def fake_embed(text):
        return [0.1] * 768

    async def fake_search(self, *args, **kwargs):
        return search_results

    async def fake_generate(model, prompt):
        return answer

    monkeypatch.setattr(app_module.embedding_provider, "embed", fake_embed)
    monkeypatch.setattr(QdrantClient, "similarity_search", fake_search)
    monkeypatch.setattr(app_module.llm_client, "generate", fake_generate)


def test_documents_query_no_hits_returns_not_found(monkeypatch):
    """/documents/query with no search hits says the answer is not in the document."""
    _patch_query_dependencies(monkeypatch, [], "not found in the document")

    response = client.post("/documents/query", json={"question": "What is the key finding?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "not found in the document"
    assert data["sources"] == []


def test_documents_query_with_results(monkeypatch):
    """/documents/query returns an answer and its sources when hits exist."""
    metadata = VectorStoreMetadata(
        id="doc-1",
        content="Key finding about AI",
        source_document="test.pdf",
        page_number=1,
        chunk_index=0,
    )
    hit = VectorStoreSearchResult(vector_id="id-1", score=0.95, metadata=metadata)
    _patch_query_dependencies(monkeypatch, [hit], "The key finding is about AI.")

    response = client.post(
        "/documents/query", json={"question": "What is the key finding?", "top_k": 4}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    source = data["sources"][0]
    assert source["document_id"] == "test.pdf"
    assert source["page"] == 1
    assert source["chunk_index"] == 0
    assert "AI" in source["text_preview"]

def test_documents_analyze_returns_summary(monkeypatch):
    monkeypatch.setattr(app_module, "extract_text", lambda data: "Report text")
    monkeypatch.setattr(app_module.audit, "record", lambda *args, **kwargs: None)

    async def fake_generate(model, prompt):
        return "A short summary."

    monkeypatch.setattr(app_module.llm_client, "generate", fake_generate)

    response = client.post(
        "/documents/analyze",
        files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "A short summary."
    assert data["extracted_text"] == "Report text"


def test_documents_analyze_invalid_pdf_returns_422(monkeypatch):
    def raise_invalid(data):
        raise InvalidPDFError("Failed to open PDF")

    monkeypatch.setattr(app_module, "extract_text", raise_invalid)

    response = client.post(
        "/documents/analyze",
        files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 422