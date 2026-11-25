from __future__ import annotations

import json

import httpx
import pytest

from app.ai._vector_store import VectorStoreMetadata
from app.ai.vector_store_qdrant import QdrantClient
from app.errors import (
    CollectionNotFoundError,
    UpstreamResponseError,
    VectorStoreUnavailableError,
)


@pytest.mark.asyncio
async def test_upsert_uses_put_wait_true_with_payload():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["params"] = dict(request.url.params)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": {"status": "completed"}})

    client = QdrantClient(transport=httpx.MockTransport(handler))
    await client.upsert_points(
        "docs",
        [
            {
                "id": "4d6a3f0e-6d3b-4a4f-9a2d-2c1f2b3c4d5e",
                "vector": [0.1, 0.2, 0.3],
                "payload": {"content": "hello", "document_id": "doc-1"},
            }
        ],
    )
    assert captured["method"] == "PUT"
    assert captured["params"] == {"wait": "true"}
    assert captured["body"]["points"][0]["payload"]["document_id"] == "doc-1"


@pytest.mark.asyncio
async def test_insert_replaces_invalid_id_with_uuid():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"result": {}})

    client = QdrantClient(transport=httpx.MockTransport(handler))
    metadata = VectorStoreMetadata(id="not-a-uuid", content="text", source_document="a.pdf")
    point_id = await client.insert("docs", [0.1, 0.2], metadata)
    assert captured["points"][0]["id"] == point_id
    assert captured["points"][0]["payload"]["id"] == point_id


@pytest.mark.asyncio
async def test_search_uses_query_api_and_parses_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/collections/docs/points/query"
        assert body["filter"] == {"must": [{"key": "document_id", "match": {"value": "doc-1"}}]}
        return httpx.Response(
            200,
            json={
                "result": {
                    "points": [
                        {
                            "id": "chunk-uuid",
                            "score": 0.91,
                            "payload": {
                                "id": "chunk-uuid",
                                "content": "retrieved text",
                                "source_document": "manual.pdf",
                                "page_number": 2,
                                "chunk_index": 3,
                                "document_id": "doc-1",
                                "chunk_id": "chunk-uuid",
                                "language": "en",
                            },
                        }
                    ]
                }
            },
        )

    client = QdrantClient(transport=httpx.MockTransport(handler))
    results = await client.similarity_search(
        "docs", [0.1, 0.2], top_k=4, filter_metadata={"document_id": "doc-1"}
    )
    assert len(results) == 1
    result = results[0]
    assert result.score == pytest.approx(0.91)
    assert result.metadata.document_id == "doc-1"
    assert result.metadata.source_document == "manual.pdf"
    assert result.metadata.page_number == 2
    assert result.metadata.extra["language"] == "en"


@pytest.mark.asyncio
async def test_search_accepts_legacy_list_response():
    client = QdrantClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": "legacy",
                            "score": 0.5,
                            "payload": {"content": "legacy text", "source_document": "old.pdf"},
                        }
                    ]
                },
            )
        )
    )
    results = await client.similarity_search("docs", [0.1], top_k=1)
    assert results[0].metadata.chunk_id == "legacy"
    assert results[0].metadata.content == "legacy text"


@pytest.mark.asyncio
async def test_create_collection_treats_409_as_success():
    client = QdrantClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(409, json={"error": "already exists"})
        )
    )
    await client.create_collection("docs", 768)


@pytest.mark.asyncio
async def test_create_collection_propagates_bad_request():
    client = QdrantClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(400, json={"error": "bad"})
        )
    )
    with pytest.raises(UpstreamResponseError):
        await client.create_collection("docs", 768)


@pytest.mark.asyncio
async def test_missing_collection_raises_domain_error():
    client = QdrantClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(404, json={"error": "not found"})
        )
    )
    with pytest.raises(CollectionNotFoundError):
        await client.get_collection_info("missing")


@pytest.mark.asyncio
async def test_connection_error_maps_to_unavailable():
    def blocked(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = QdrantClient(transport=httpx.MockTransport(blocked))
    with pytest.raises(VectorStoreUnavailableError):
        await client.get_collection_info("docs")
    health = await client.health()
    assert health["status"] == "unhealthy"


def test_collection_info_helpers():
    info = {
        "config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}},
        "points_count": 12,
    }
    assert QdrantClient.vector_size_from_info(info) == 768
    assert QdrantClient.point_count_from_info(info) == 12
    named = {"config": {"params": {"vectors": {"text": {"size": 384}}}}}
    assert QdrantClient.vector_size_from_info(named) == 384


@pytest.mark.asyncio
async def test_delete_points_uses_batch_endpoint():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": {}})

    client = QdrantClient(transport=httpx.MockTransport(handler))
    await client.delete_points("docs", ["a", "b"])
    assert captured["path"] == "/collections/docs/points/delete"
    assert captured["body"] == {"points": ["a", "b"]}
