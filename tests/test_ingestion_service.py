from __future__ import annotations

import pytest

from app.errors import (
    CollectionCompatibilityError,
    DocumentTooLargeToIndexError,
    EmbeddingUnavailableError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)


def _prepare(container):
    container.audit.initialize()
    container.registry.initialize()
    return container.ingestion_service


@pytest.mark.asyncio
async def test_ingest_text_document_stores_chunks(container):
    service = _prepare(container)
    result = await service.ingest(
        "handbook.txt",
        "text/plain",
        b"Paris is the capital of France. Bread is baked in an oven.",
    )
    assert result.duplicate is False
    assert result.record.chunk_count > 0
    assert result.record.filename == "handbook.txt"
    points = container.vector_store.collections["documents"]["points"]
    assert len(points) == result.record.chunk_count
    payload = next(iter(points.values()))["payload"]
    assert payload["document_id"] == result.record.document_id
    assert payload["source_document"] == "handbook.txt"
    assert payload["chunk_id"]


@pytest.mark.asyncio
async def test_ingest_is_idempotent_for_duplicate_content(container):
    service = _prepare(container)
    data = b"Duplicate content that should only be indexed once."
    first = await service.ingest("a.txt", "text/plain", data)
    second = await service.ingest("b.txt", "text/plain", data)
    assert first.duplicate is False
    assert second.duplicate is True
    assert second.record.document_id == first.record.document_id
    assert len(container.vector_store.collections["documents"]["points"]) == first.record.chunk_count


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_file(container):
    service = _prepare(container)
    with pytest.raises(UnsupportedFileTypeError):
        await service.ingest("image.png", "image/png", b"not supported")


@pytest.mark.asyncio
async def test_ingest_rejects_empty_document(container):
    service = _prepare(container)
    with pytest.raises(NoExtractableTextError):
        await service.ingest("blank.txt", "text/plain", b"   \n\n   ")


@pytest.mark.asyncio
async def test_ingest_fails_when_embedding_service_is_down(container):
    service = _prepare(container)
    container.embedding_provider.unavailable = True
    with pytest.raises(EmbeddingUnavailableError):
        await service.ingest("doc.txt", "text/plain", b"Some usable content here.")


@pytest.mark.asyncio
async def test_ingest_cleans_up_points_when_upsert_fails(container):
    service = _prepare(container)
    container.vector_store.fail_upsert = True
    with pytest.raises(Exception):
        await service.ingest("doc.txt", "text/plain", b"Some usable content here.")
    assert container.vector_store.deleted_points
    assert container.registry.count() == 0


@pytest.mark.asyncio
async def test_ingest_detects_collection_dimension_mismatch(container):
    service = _prepare(container)
    await container.vector_store.create_collection("documents", 8)
    with pytest.raises(CollectionCompatibilityError):
        await service.ingest("doc.txt", "text/plain", b"Some usable content here.")


@pytest.mark.asyncio
async def test_ingest_detects_embedding_model_change(container):
    service = _prepare(container)
    await container.vector_store.create_collection("documents", container.embedding_provider.dimension)
    container.registry.set_collection_state("documents", "old-model", container.embedding_provider.dimension)
    with pytest.raises(CollectionCompatibilityError):
        await service.ingest("doc.txt", "text/plain", b"Some usable content here.")


@pytest.mark.asyncio
async def test_ingest_enforces_chunk_limit(container):
    service = _prepare(container)
    container.settings.max_chunks_per_document = 1
    with pytest.raises(DocumentTooLargeToIndexError):
        await service.ingest("long.txt", "text/plain", ("word " * 500).encode("utf-8"))
