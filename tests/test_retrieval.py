from __future__ import annotations

import pytest

from app.rag.retrieval import RetrievalService


async def _seed(vector_store, embedding, collection: str) -> None:
    await vector_store.create_collection(collection, embedding.dimension)
    texts = [
        ("chunk-1", "The capital of France is Paris.", "doc-1", 1, 0),
        ("chunk-2", "Bread is baked in an oven.", "doc-2", 1, 0),
    ]
    points = []
    for chunk_id, text, document_id, page, chunk_index in texts:
        points.append(
            {
                "id": chunk_id,
                "vector": await embedding.embed(text),
                "payload": {
                    "id": chunk_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source_document": f"{document_id}.pdf",
                    "page_number": page,
                    "chunk_index": chunk_index,
                    "content": text,
                },
            }
        )
    await vector_store.upsert_points(collection, points)


@pytest.mark.asyncio
async def test_retrieval_returns_relevant_chunks_with_metadata(fakes):
    embedding, vector_store, _ = fakes
    await _seed(vector_store, embedding, "documents")
    service = RetrievalService(embedding, vector_store, "documents", score_threshold=0.0)
    chunks = await service.retrieve("What is the capital of France?", top_k=2)
    assert chunks
    assert chunks[0].content == "The capital of France is Paris."
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].document_name == "doc-1.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].citation_id == 1


@pytest.mark.asyncio
async def test_retrieval_applies_document_filter(fakes):
    embedding, vector_store, _ = fakes
    await _seed(vector_store, embedding, "documents")
    service = RetrievalService(embedding, vector_store, "documents", score_threshold=0.0)
    chunks = await service.retrieve("capital France bread oven", document_id="doc-2")
    assert chunks
    assert all(chunk.document_id == "doc-2" for chunk in chunks)


@pytest.mark.asyncio
async def test_retrieval_applies_score_threshold(fakes):
    embedding, vector_store, _ = fakes
    await _seed(vector_store, embedding, "documents")
    service = RetrievalService(embedding, vector_store, "documents", score_threshold=0.99)
    chunks = await service.retrieve("The capital of France is Paris.")
    assert [chunk.content for chunk in chunks] == ["The capital of France is Paris."]


@pytest.mark.asyncio
async def test_retrieval_without_collection_returns_empty(fakes):
    embedding, vector_store, _ = fakes
    service = RetrievalService(embedding, vector_store, "documents")
    assert await service.retrieve("anything") == []


@pytest.mark.asyncio
async def test_retrieval_rejects_empty_question(fakes):
    embedding, vector_store, _ = fakes
    await _seed(vector_store, embedding, "documents")
    service = RetrievalService(embedding, vector_store, "documents")
    assert await service.retrieve("   ") == []
