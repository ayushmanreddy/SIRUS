from __future__ import annotations

import json

import httpx
import pytest

from app.ai.embedding_ollama import OllamaEmbeddingProvider
from app.errors import EmbeddingResponseError, EmbeddingUnavailableError


def _blocked(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


@pytest.mark.asyncio
async def test_embed_uses_api_embed_with_right_body():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert body["model"] == "nomic-embed-text"
        assert body["input"] == "hello world"
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4]]})

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        model_name="nomic-embed-text",
        transport=httpx.MockTransport(handler),
    )
    assert await provider.embed("hello world") == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_embed_batch_sends_list_of_texts():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == ["text1", "text2"]
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        model_name="nomic-embed-text",
        transport=httpx.MockTransport(handler),
    )
    results = await provider.embed_batch(["text1", "text2"])
    assert results == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_embed_dimension_mismatch_raises():
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        model_name="nomic-embed-text",
        dimensions=16,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
        ),
    )
    with pytest.raises(EmbeddingResponseError, match="dimension"):
        await provider.embed("test")


@pytest.mark.asyncio
async def test_embed_rejects_empty_and_non_finite_vectors():
    empty_provider = OllamaEmbeddingProvider(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": [[]]})
        )
    )
    with pytest.raises(EmbeddingResponseError):
        await empty_provider.embed("test")

    bad_provider = OllamaEmbeddingProvider(
        model_name="m",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": [[0.1, float("nan")]]})
        ),
    )
    with pytest.raises(EmbeddingResponseError):
        await bad_provider.embed("test")


@pytest.mark.asyncio
async def test_batch_count_mismatch_raises():
    provider = OllamaEmbeddingProvider(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        )
    )
    with pytest.raises(EmbeddingResponseError, match="count"):
        await provider.embed_batch(["one", "two"])


@pytest.mark.asyncio
async def test_embedding_service_unavailable_is_wrapped():
    provider = OllamaEmbeddingProvider(transport=httpx.MockTransport(_blocked))
    with pytest.raises(EmbeddingUnavailableError):
        await provider.embed("test")


@pytest.mark.asyncio
async def test_get_dimension_does_not_silently_fallback():
    provider = OllamaEmbeddingProvider(transport=httpx.MockTransport(_blocked))
    with pytest.raises(EmbeddingUnavailableError):
        await provider.get_dimension()


@pytest.mark.asyncio
async def test_health_reports_missing_model():
    provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
        ),
    )
    health = await provider.health()
    assert health["status"] == "unhealthy"
    assert "not installed" in health["detail"]


@pytest.mark.asyncio
async def test_health_accepts_latest_tag():
    provider = OllamaEmbeddingProvider(
        model_name="nomic-embed-text",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"models": [{"name": "nomic-embed-text:latest"}]}
            )
        ),
    )
    health = await provider.health()
    assert health["status"] == "healthy"
