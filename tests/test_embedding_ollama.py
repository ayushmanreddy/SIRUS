import json

import pytest

import httpx

from app.ai.embedding_ollama import OllamaEmbeddingProvider


@pytest.mark.asyncio
async def test_embed_uses_api_embed_with_right_body():
    """embed calls POST /api/embed with payload {"model": ..., "input": text}."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "nomic-embed-text"
        assert body["input"] == "hello world"
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]},
        )

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model_name="nomic-embed-text",
        transport=httpx.MockTransport(handler)
    )

    result = await provider.embed("hello world")
    assert result == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


@pytest.mark.asyncio
async def test_embed_batch_sends_list_of_texts():
    """embed_batch sends all texts in one POST /api/embed request."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert isinstance(body["input"], list)
        assert body["input"] == ["text1", "text2"]
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model_name="nomic-embed-text",
        transport=httpx.MockTransport(handler)
    )

    results = await provider.embed_batch(["text1", "text2"])
    assert len(results) == 2
    assert results[0] == [0.1, 0.2]
    assert results[1] == [0.3, 0.4]


@pytest.mark.asyncio
async def test_embed_dimension_mismatch_raises():
    """embed raises ValueError when returned vector length differs from configured dimensions."""

    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model_name="nomic-embed-text", dimensions=16,
        transport=httpx.MockTransport(lambda r: httpx.Response(
            200, json={"embeddings": [[0.1, 0.2, 0.3]]}
        ))
    )

    try:
        await provider.embed("test")
        pytest.fail("Should have raised ValueError for dimension mismatch")
    except ValueError as e:
        assert "dimension" in str(e).lower()