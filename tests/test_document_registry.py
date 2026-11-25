from __future__ import annotations

import sqlite3

import pytest

from app.services.document_registry import DocumentRecord, DocumentRegistry


def _registry(tmp_path) -> DocumentRegistry:
    registry = DocumentRegistry(f"sqlite:///{(tmp_path / 'registry.db').as_posix()}")
    registry.initialize()
    return registry


def _record(**overrides) -> DocumentRecord:
    values = {
        "document_id": "doc-1",
        "filename": "manual.pdf",
        "content_type": "application/pdf",
        "sha256": "a" * 64,
        "size_bytes": 100,
        "page_count": 2,
        "chunk_count": 3,
        "embedding_model": "nomic-embed-text",
        "embedding_dimension": 768,
        "collection_name": "documents",
        "created_at": "2026-01-01T00:00:00+00:00",
        "chunks": [{"chunk_id": "chunk-1", "page": 1, "chunk_index": 0, "text_preview": "x"}],
    }
    values.update(overrides)
    return DocumentRecord(**values)


def test_add_and_get_document_roundtrip(tmp_path):
    registry = _registry(tmp_path)
    registry.add(_record())
    loaded = registry.get("doc-1")
    assert loaded is not None
    assert loaded.filename == "manual.pdf"
    assert loaded.chunks[0]["chunk_id"] == "chunk-1"


def test_duplicate_detection_by_hash_model_and_collection(tmp_path):
    registry = _registry(tmp_path)
    registry.add(_record())
    assert registry.find_duplicate("a" * 64, "nomic-embed-text", "documents") is not None
    assert registry.find_duplicate("a" * 64, "other-model", "documents") is None
    assert registry.find_duplicate("b" * 64, "nomic-embed-text", "documents") is None


def test_duplicate_insert_raises_integrity_error(tmp_path):
    registry = _registry(tmp_path)
    registry.add(_record())
    with pytest.raises(sqlite3.IntegrityError):
        registry.add(_record(document_id="doc-2"))


def test_list_documents_and_count(tmp_path):
    registry = _registry(tmp_path)
    registry.add(_record())
    registry.add(_record(document_id="doc-2", sha256="b" * 64, filename="b.txt"))
    documents, total = registry.list_documents(limit=10, offset=0)
    assert total == 2
    assert {record.document_id for record in documents} == {"doc-1", "doc-2"}
    limited, _ = registry.list_documents(limit=1, offset=1)
    assert len(limited) == 1
    assert registry.count() == 2


def test_collection_state_upsert(tmp_path):
    registry = _registry(tmp_path)
    assert registry.get_collection_state("documents") is None
    registry.set_collection_state("documents", "nomic-embed-text", 768)
    state = registry.get_collection_state("documents")
    assert state is not None
    assert state.embedding_model == "nomic-embed-text"
    registry.set_collection_state("documents", "other-model", 384)
    updated = registry.get_collection_state("documents")
    assert updated is not None
    assert updated.embedding_model == "other-model"
    assert updated.embedding_dimension == 384
