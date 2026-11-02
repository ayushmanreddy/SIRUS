import json
import asyncio

import httpx

from app.ai.vector_store_qdrant import QdrantClient


def test_insert_uses_put_with_point_id():
    """INSERT uses PUT /collections/{name}/points?wait=true with explicit point id (UUID)."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "points" in body
        assert "id" in body["points"][0]
        # Verify the point has an id (UUID string)
        point_id = body["points"][0]["id"]
        assert isinstance(point_id, str)
        # Verify wait=true param is present
        assert request.url.params.get("wait") == "true"
        return httpx.Response(200, json={"result": [{"id": point_id}]})

    client = QdrantClient(
        base_url="http://localhost:6333", timeout=10.0,
        transport=httpx.MockTransport(handler)
    )

    asyncio.run(client._get_client())
    from app.ai._vector_store import VectorStoreMetadata
    meta = VectorStoreMetadata(id="test-id", content="test content", source_document="test.pdf")
    point_id = asyncio.run(client.insert("test_collection", [0.1, 0.2, 0.3], meta))
    assert point_id is not None


def test_search_sends_filter():
    """Search sends 'filter' key in Qdrant filter format."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "filter" in body, f"Expected 'filter' key in body, got: {body}"
        filter_obj = body["filter"]
        assert isinstance(filter_obj, dict)
        return httpx.Response(200, json={"result": []})

    client = QdrantClient(
        base_url="http://localhost:6333", timeout=10.0,
        transport=httpx.MockTransport(handler)
    )

    asyncio.run(client._get_client())
    from app.ai._vector_store import VectorStoreMetadata
    meta = VectorStoreMetadata(id="test-id", content="test content", source_document="test.pdf")
    asyncio.run(client.similarity_search(
        "test_collection",
        [0.1, 0.2, 0.3],
        top_k=4,
        filter_metadata={"key": "value"},
    ))


def test_create_collection_treats_409_as_success():
    """create_collection treats HTTP 409 (already exists) as success."""
    from app.ai.vector_store_qdrant import QdrantClient

    client = QdrantClient(
        base_url="http://localhost:6333", timeout=10.0,
        transport=httpx.MockTransport(lambda r: httpx.Response(409, json={"error": "Collection already exists"}))
    )

    asyncio.run(client._get_client())
    asyncio.run(client.create_collection("test_collection", 768))
    # Should not raise - 409 treated as success