"""Vector store abstraction for SIRUS.

The interface keeps the RAG pipeline independent of the Qdrant REST details.
Qdrant is the only V1 implementation; future local vector stores can implement
the same contract without rewriting ingestion or retrieval.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sirus.vector_store")


class VectorStoreMetadata:
    """Metadata associated with a vector in the store."""

    def __init__(
        self,
        id: str,
        content: str,
        source_document: str,
        page_number: int | None = None,
        chunk_index: int | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
        **extra: Any,
    ) -> None:
        self.id = id
        self.content = content
        self.source_document = source_document
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.extra = extra

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "content": self.content,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            **self.extra,
        }
        return {key: value for key, value in payload.items() if value is not None}


class VectorStoreSearchResult:
    """Result of a vector similarity search."""

    def __init__(
        self,
        vector_id: str,
        score: float,
        metadata: VectorStoreMetadata,
    ) -> None:
        self.vector_id = vector_id
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "score": self.score,
            "metadata": self.metadata.to_dict(),
        }


class VectorStoreInterface(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    async def health(self) -> Dict[str, Any]:
        """Check vector store health status."""

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata_schema: Dict[str, Any] | None = None,
    ) -> None:
        """Create a new vector collection."""

    @abstractmethod
    async def insert(
        self,
        collection_name: str,
        vector: List[float],
        metadata: VectorStoreMetadata,
    ) -> str:
        """Insert a vector into a collection. Returns the vector ID."""

    @abstractmethod
    async def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
    ) -> None:
        """Insert or replace points in one or more batched requests."""

    @abstractmethod
    async def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 4,
        filter_metadata: Dict[str, Any] | None = None,
    ) -> List[VectorStoreSearchResult]:
        """Perform similarity search over vectors in a collection."""

    @abstractmethod
    async def delete(
        self,
        collection_name: str,
        vector_id: str,
    ) -> None:
        """Delete a vector from a collection."""

    @abstractmethod
    async def delete_points(self, collection_name: str, vector_ids: List[str]) -> None:
        """Delete several vectors from a collection."""

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection."""

    @abstractmethod
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the vector store."""

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a specific collection."""

    async def close(self) -> None:
        """Release network resources. Implementations may override."""
