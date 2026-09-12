from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from app.ai._vector_store import (
    VectorStoreInterface,
    VectorStoreMetadata,
    VectorStoreSearchResult,
)

logger = logging.getLogger("sovereign_workbench.vector_store.qdrant")


class QdrantClient(VectorStoreInterface):
    """Qdrant vector store client implementing the VectorStoreInterface.

    Connects to a local or Qdrant Compose service instance. All operations
    stay within the local deployment boundary when configured for offline‑demo
    mode (loopback/private URLs only).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:6333",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def health(self) -> Dict[str, Any]:
        """Check Qdrant service health."""
        try:
            client = await self._get_client()
            response = await client.get("/healthz")
            if response.status_code == 200:
                return {"status": "healthy", "service": "qdrant", "url": self.base_url}
            return {"status": "unhealthy", "service": "qdrant", "url": self.base_url, "detail": f"status {response.status_code}"}
        except (httpx.HTTPError, ConnectionError) as error:
            logger.warning("Qdrant health check failed: %s", error)
            return {"status": "unhealthy", "service": "qdrant", "url": self.base_url, "detail": str(error)}

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new vector collection in Qdrant."""
        try:
            client = await self._get_client()
            payload: Dict[str, Any] = {
                "vectors": {"size": dimension, "distance": "Cosine"},
            }
            if metadata_schema:
                payload["metadata_schema"] = metadata_schema
            await client.put(f"/collections/{collection_name}", json=payload)
            logger.info("Qdrant collection created: %s", collection_name)
        except (httpx.HTTPError, ConnectionError) as error:
            logger.error("Failed to create Qdrant collection: %s", error)
            raise

    async def insert(
        self,
        collection_name: str,
        vector: List[float],
        metadata: VectorStoreMetadata,
    ) -> str:
        """Insert a vector into a Qdrant collection."""
        try:
            client = await self._get_client()
            payload = {
                "vector": vector,
                "payload": metadata.to_dict(),
            }
            await client.post(f"/collections/{collection_name}/points", json=payload)
            logger.info("Qdrant vector inserted to collection %s", collection_name)
            return metadata.id
        except (httpx.HTTPError, ConnectionError) as error:
            logger.error("Failed to insert Qdrant vector: %s", error)
            raise

    async def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        """Perform similarity search over Qdrant vectors."""
        try:
            client = await self._get_client()
            payload: Dict[str, Any] = {
                "query_vector": query_vector,
                "limit": top_k,
                "with_payload": True,
            }
            if filter_metadata:
                # Build a Qdrant filter from the metadata dict
                # Simplified: just pass as query parameters
                payload["metadata_filter"] = filter_metadata

            response = await client.post(
                f"/collections/{collection_name}/points/search",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            results: List[VectorStoreSearchResult] = []
            for point in data.get("result", []):
                score = point.get("score", 0.0)
                payload = point.get("payload", {})
                metadata = VectorStoreMetadata(
                    id=point.get("id", ""),
                    content=payload.get("content", ""),
                    source_document=payload.get("source_document", ""),
                    page_number=payload.get("page_number"),
                    chunk_index=payload.get("chunk_index"),
                    **{k: v for k, v in payload.items() if k not in ("content", "source_document", "page_number", "chunk_index")},
                )
                results.append(VectorStoreSearchResult(vector_id=point.get("id", ""), score=score, metadata=metadata))
            return results
        except (httpx.HTTPError, ConnectionError) as error:
            logger.error("Qdrant similarity search failed: %s", error)
            raise

    async def delete(self, collection_name: str, vector_id: str) -> None:
        """Delete a vector from a Qdrant collection."""
        try:
            client = await self._get_client()
            await client.delete(f"/collections/{collection_name}/points/{vector_id}")
            logger.info("Qdrant vector deleted: %s from collection %s", vector_id, collection_name)
        except (httpx.HTTPError, ConnectionError) as error:
            logger.error("Failed to delete Qdrant vector: %s", error)
            raise

    async def delete_collection(self, collection_name: str) -> None:
        """Delete an entire Qdrant collection."""
        try:
            client = await self._get_client()
            await client.delete(f"/collections/{collection_name}")
            logger.info("Qdrant collection deleted: %s", collection_name)
        except (httpx.HTTPError, ConnectionError) as error:
            logger.error("Failed to delete Qdrant collection: %s", error)
            raise

    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all Qdrant collections."""
        try:
            client = await self._get_client()
            response = await client.get("/collections")
            response.raise_for_status()
            data = response.json()
            return data.get("result", [])
        except (httpx.HTTPError, ConnectionError) as error:
            logger.error("Failed to list Qdrant collections: %s", error)
            return []

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get Qdrant collection information."""
        try:
            client = await self._get_client()
            response = await client.get(f"/collections/{collection_name}")
            response.raise_for_status()
            return response.json().get("result", {})
        except (httpx.HTTPError, ConnectionError) as error:
            logger.error("Failed to get Qdrant collection info: %s", error)
            return {}