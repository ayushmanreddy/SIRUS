"""Vector store abstraction for the Sovereign AI Workbench.

Defines the interface for local vector database operations. Supports
collections, vector insertion, similarity search, metadata, deletion,
and persistence. Designed to be implemented by Qdrant (current) or
future local vector stores.

The abstraction ensures the application is not tightly coupled to one
vector database technology, allowing future replacement without rewriting
the entire RAG pipeline.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sovereign_workbench.vector_store")


class VectorStoreMetadata:
    """Metadata associated with a vector in the store."""

    def __init__(
        self,
        id: str,
        content: str,
        source_document: str,
        page_number: int | None = None,
        chunk_index: int | None = None,
        **extra: Any,
    ) -> None:
        self.id = id
        self.content = content
        self.source_document = source_document
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.extra = extra

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            **self.extra,
        }


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
    async def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection."""

    @abstractmethod
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the vector store."""

    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a specific collection."""