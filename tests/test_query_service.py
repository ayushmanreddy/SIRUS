from __future__ import annotations

import pytest

from app.errors import DocumentNotFoundError
from app.llm_client import ModelUnavailableError
from app.rag.prompts import NO_EVIDENCE_ANSWER
from app.rag.retrieval import RetrievalService
from app.services.query import QueryService


def _prepare(container) -> None:
    container.audit.initialize()
    container.registry.initialize()


async def _ingest(container) -> str:
    result = await container.ingestion_service.ingest(
        "france.txt",
        "text/plain",
        b"Paris is the capital of France. The Eiffel Tower is in Paris.",
    )
    return result.record.document_id


def _service_with_threshold(container, threshold: float) -> QueryService:
    retrieval = RetrievalService(
        embedding_provider=container.embedding_provider,
        vector_store=container.vector_store,
        collection_name=container.settings.qdrant_collection,
        default_top_k=container.settings.retrieval_top_k,
        score_threshold=threshold,
    )
    return QueryService(
        settings=container.settings,
        registry=container.registry,
        retrieval=retrieval,
        vector_store=container.vector_store,
        llm_client=container.llm_client,
        audit=container.audit,
    )


@pytest.mark.asyncio
async def test_query_answers_from_seeded_document(container):
    _prepare(container)
    document_id = await _ingest(container)
    result = await container.query_service.answer("What is the capital of France?")
    assert result.grounded is True
    assert result.sources
    assert result.sources[0].document_id == document_id
    assert result.sources[0].document_name == "france.txt"
    assert result.sources[0].citation_id == 1
    assert "system" in container.llm_client.calls[0]


@pytest.mark.asyncio
async def test_query_before_any_documents_returns_honest_answer(container):
    _prepare(container)
    result = await container.query_service.answer("What is the capital of France?")
    assert result.grounded is False
    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.sources == []
    assert container.embedding_provider.embed_calls == 0
    assert container.llm_client.calls == []


@pytest.mark.asyncio
async def test_query_with_no_chunks_above_threshold_returns_honest_answer(container):
    _prepare(container)
    await _ingest(container)
    high_threshold_service = _service_with_threshold(container, 0.99)
    result = await high_threshold_service.answer("Completely unrelated xylophone question")
    assert result.grounded is False
    assert result.sources == []
    assert result.answer == NO_EVIDENCE_ANSWER
    assert container.llm_client.calls == []


@pytest.mark.asyncio
async def test_query_with_unknown_document_id_raises_not_found(container):
    _prepare(container)
    await _ingest(container)
    with pytest.raises(DocumentNotFoundError):
        await container.query_service.answer("capital", document_id="missing-doc")


@pytest.mark.asyncio
async def test_query_filters_by_document_id(container):
    _prepare(container)
    document_id = await _ingest(container)
    await container.ingestion_service.ingest("other.txt", "text/plain", b"Unrelated zebra content.")
    result = await container.query_service.answer("capital of France", document_id=document_id)
    assert result.sources
    assert {source.document_id for source in result.sources} == {document_id}


@pytest.mark.asyncio
async def test_query_fails_when_llm_is_unavailable(container):
    _prepare(container)
    await _ingest(container)
    container.llm_client.unavailable = True
    with pytest.raises(ModelUnavailableError):
        await container.query_service.answer("What is the capital of France?")


@pytest.mark.asyncio
async def test_query_normalizes_model_insufficient_answer(container):
    _prepare(container)
    await _ingest(container)
    container.llm_client.answer = "I cannot find enough information in the provided context."
    result = await container.query_service.answer("What is the capital of France?")
    assert result.grounded is False
    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.sources
