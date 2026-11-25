from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import httpx

from app.ai._vector_store import (
    VectorStoreInterface,
    VectorStoreMetadata,
    VectorStoreSearchResult,
)
from app.errors import (
    CollectionNotFoundError,
    UpstreamResponseError,
    VectorStoreUnavailableError,
)

logger = logging.getLogger("sirus.vector_store.qdrant")

_UPSERT_BATCH_SIZE = 128
_KNOWN_PAYLOAD_KEYS = {
    "id",
    "content",
    "source_document",
    "page_number",
    "chunk_index",
    "document_id",
    "chunk_id",
}


class QdrantClient(VectorStoreInterface):
    """Qdrant REST client used by the local SIRUS deployment."""

    def __init__(
        self,
        base_url: str = "http://localhost:6333",
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._transport = transport

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            if self._transport:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url, timeout=self.timeout, transport=self._transport
                )
            else:
                self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: Dict[str, Any] | None = None,
        allow_404: bool = False,
        allow_409: bool = False,
    ) -> httpx.Response:
        client = await self._get_client()
        try:
            response = await client.request(method, path, json=json, params=params)
        except (httpx.HTTPError, OSError) as error:
            logger.error("Qdrant request %s %s failed: %s", method, path, error)
            raise VectorStoreUnavailableError("The vector store is not reachable.") from error

        if response.status_code == 404 and allow_404:
            return response
        if response.status_code == 409 and allow_409:
            return response
        if response.status_code == 404:
            raise CollectionNotFoundError()
        if response.is_error:
            detail = response.text[:300] if response.text else "no response body"
            logger.error(
                "Qdrant request %s %s returned %s: %s",
                method,
                path,
                response.status_code,
                detail,
            )
            raise UpstreamResponseError("The vector store rejected the request.")
        return response

    async def health(self) -> Dict[str, Any]:
        try:
            client = await self._get_client()
            response = await client.get("/healthz")
            if response.status_code == 200:
                return {"status": "healthy", "service": "qdrant"}
            return {"status": "unhealthy", "service": "qdrant", "detail": f"status {response.status_code}"}
        except (httpx.HTTPError, OSError) as error:
            logger.warning("Qdrant health check failed: %s", error)
            return {"status": "unhealthy", "service": "qdrant", "detail": "service not reachable"}

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("collection dimension must be positive")
        payload: Dict[str, Any] = {"vectors": {"size": dimension, "distance": "Cosine"}}
        if metadata_schema:
            payload["metadata_schema"] = metadata_schema
        response = await self._request(
            "PUT",
            f"/collections/{collection_name}",
            json=payload,
            allow_409=True,
        )
        if response.status_code == 409:
            logger.info("Qdrant collection already exists: %s", collection_name)
        else:
            logger.info("Qdrant collection created: %s (dimension=%s)", collection_name, dimension)

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        response = await self._request("GET", f"/collections/{collection_name}")
        data = response.json()
        return data.get("result", {}) if isinstance(data, dict) else {}

    @staticmethod
    def vector_size_from_info(info: Dict[str, Any]) -> int | None:
        """Extract the configured vector size from a Qdrant collection info payload."""
        vectors = (((info or {}).get("config") or {}).get("params") or {}).get("vectors")
        if isinstance(vectors, dict):
            if isinstance(vectors.get("size"), int):
                return vectors["size"]
            for value in vectors.values():
                if isinstance(value, dict) and isinstance(value.get("size"), int):
                    return value["size"]
        return None

    @staticmethod
    def point_count_from_info(info: Dict[str, Any]) -> int | None:
        count = (info or {}).get("points_count")
        return count if isinstance(count, int) else None

    async def insert(
        self,
        collection_name: str,
        vector: List[float],
        metadata: VectorStoreMetadata,
    ) -> str:
        try:
            point_id = str(uuid.UUID(str(metadata.id)))
        except (ValueError, AttributeError, TypeError):
            point_id = str(uuid.uuid4())
        payload = metadata.to_dict()
        payload["id"] = point_id
        await self.upsert_points(
            collection_name,
            [{"id": point_id, "vector": vector, "payload": payload}],
        )
        return point_id

    async def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
    ) -> None:
        if not points:
            return
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[start : start + _UPSERT_BATCH_SIZE]
            await self._request(
                "PUT",
                f"/collections/{collection_name}/points",
                json={"points": batch},
                params={"wait": "true"},
            )
        logger.info("Qdrant upserted %s points into %s", len(points), collection_name)

    async def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        if not query_vector:
            raise ValueError("query vector cannot be empty")
        payload: Dict[str, Any] = {
            "query": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        if filter_metadata:
            payload["filter"] = {
                "must": [
                    {
                        "key": key,
                        "match": {
                            "value": value
                            if isinstance(value, (str, int, float, bool))
                            else str(value)
                        },
                    }
                    for key, value in filter_metadata.items()
                    if value is not None
                ]
            }

        response = await self._request(
            "POST",
            f"/collections/{collection_name}/points/query",
            json=payload,
        )
        data = response.json()
        result = data.get("result", {}) if isinstance(data, dict) else {}
        raw_points = result.get("points", []) if isinstance(result, dict) else result

        results: List[VectorStoreSearchResult] = []
        for point in raw_points or []:
            point_payload = point.get("payload") or {}
            extra = {key: value for key, value in point_payload.items() if key not in _KNOWN_PAYLOAD_KEYS}
            metadata = VectorStoreMetadata(
                id=str(point.get("id", "")),
                content=str(point_payload.get("content", "")),
                source_document=str(point_payload.get("source_document", "")),
                page_number=point_payload.get("page_number"),
                chunk_index=point_payload.get("chunk_index"),
                document_id=point_payload.get("document_id"),
                chunk_id=point_payload.get("chunk_id") or str(point.get("id", "")),
                **extra,
            )
            results.append(
                VectorStoreSearchResult(
                    vector_id=str(point.get("id", "")),
                    score=float(point.get("score", 0.0)),
                    metadata=metadata,
                )
            )
        return results

    async def delete(self, collection_name: str, vector_id: str) -> None:
        await self._request(
            "DELETE",
            f"/collections/{collection_name}/points/{vector_id}",
            params={"wait": "true"},
        )

    async def delete_points(self, collection_name: str, vector_ids: List[str]) -> None:
        if not vector_ids:
            return
        await self._request(
            "POST",
            f"/collections/{collection_name}/points/delete",
            json={"points": vector_ids},
            params={"wait": "true"},
        )

    async def delete_collection(self, collection_name: str) -> None:
        await self._request("DELETE", f"/collections/{collection_name}")

    async def list_collections(self) -> List[Dict[str, Any]]:
        response = await self._request("GET", "/collections")
        data = response.json()
        result = data.get("result", []) if isinstance(data, dict) else []
        if isinstance(result, dict):
            return result.get("collections", [])
        return result if isinstance(result, list) else []
