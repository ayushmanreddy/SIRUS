from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ai._embedding import EmbeddingProvider
from app.ai._vector_store import (
    VectorStoreInterface,
    VectorStoreMetadata,
    VectorStoreSearchResult,
)
from app.audit import AuditStore
from app.config import Settings
from app.dependencies import AppContainer
from app.errors import (
    CollectionNotFoundError,
    EmbeddingUnavailableError,
    UpstreamResponseError,
    VectorStoreUnavailableError,
)
from app.llm_client import ModelUnavailableError
from app.main import create_app
from app.rag.retrieval import RetrievalService
from app.services.document_registry import DocumentRegistry
from app.services.ingestion import IngestionService
from app.services.query import QueryService

EMBEDDING_DIMENSION = 64


def embed_text(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    """Deterministic bag-of-words hashing embedding for tests."""
    vector = [0.0] * dimension
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(word.encode("utf-8")).hexdigest()
        vector[int(digest, 16) % dimension] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "fake-embed", dimension: int = EMBEDDING_DIMENSION) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.unavailable = False
        self.embed_calls = 0
        self.batch_calls = 0

    async def health(self) -> dict[str, Any]:
        if self.unavailable:
            return {"status": "unhealthy", "detail": "embedding service unavailable"}
        return {"status": "healthy", "model": self.model_name, "available_models": [self.model_name]}

    async def embed(self, text: str) -> list[float]:
        if self.unavailable:
            raise EmbeddingUnavailableError()
        self.embed_calls += 1
        return embed_text(text, self.dimension)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.unavailable:
            raise EmbeddingUnavailableError()
        self.batch_calls += 1
        return [embed_text(text, self.dimension) for text in texts]

    async def get_dimension(self) -> int:
        if self.unavailable:
            raise EmbeddingUnavailableError()
        return self.dimension

    async def get_model_name(self) -> str:
        return self.model_name

    async def close(self) -> None:
        return None


class FakeVectorStore(VectorStoreInterface):
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.unavailable = False
        self.fail_upsert = False
        self.deleted_points: list[str] = []

    def _check(self) -> None:
        if self.unavailable:
            raise VectorStoreUnavailableError()

    async def health(self) -> dict[str, Any]:
        if self.unavailable:
            return {"status": "unhealthy", "detail": "vector store unavailable"}
        return {"status": "healthy", "service": "fake-qdrant"}

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metadata_schema: dict[str, Any] | None = None,
    ) -> None:
        self._check()
        self.collections.setdefault(collection_name, {"dimension": dimension, "points": {}})

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        self._check()
        collection = self.collections.get(collection_name)
        if collection is None:
            raise CollectionNotFoundError()
        return {
            "config": {"params": {"vectors": {"size": collection["dimension"]}}},
            "points_count": len(collection["points"]),
        }

    async def insert(
        self,
        collection_name: str,
        vector: list[float],
        metadata: VectorStoreMetadata,
    ) -> str:
        point_id = metadata.id
        await self.upsert_points(
            collection_name,
            [{"id": point_id, "vector": vector, "payload": metadata.to_dict()}],
        )
        return point_id

    async def upsert_points(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        self._check()
        if self.fail_upsert:
            raise UpstreamResponseError("vector upsert failed")
        collection = self.collections.get(collection_name)
        if collection is None:
            raise CollectionNotFoundError()
        for point in points:
            if len(point["vector"]) != collection["dimension"]:
                raise UpstreamResponseError("vector dimension mismatch")
            collection["points"][point["id"]] = {
                "vector": list(point["vector"]),
                "payload": dict(point["payload"]),
            }

    async def similarity_search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 4,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorStoreSearchResult]:
        self._check()
        collection = self.collections.get(collection_name)
        if collection is None:
            raise CollectionNotFoundError()
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for point_id, point in collection["points"].items():
            payload = point["payload"]
            if filter_metadata and any(
                payload.get(key) != value for key, value in filter_metadata.items()
            ):
                continue
            score = sum(a * b for a, b in zip(query_vector, point["vector"], strict=False))
            scored.append((score, point_id, payload))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[VectorStoreSearchResult] = []
        known = {
            "id",
            "content",
            "source_document",
            "page_number",
            "chunk_index",
            "document_id",
            "chunk_id",
        }
        for score, point_id, payload in scored[:top_k]:
            extra = {key: value for key, value in payload.items() if key not in known}
            results.append(
                VectorStoreSearchResult(
                    vector_id=point_id,
                    score=float(score),
                    metadata=VectorStoreMetadata(
                        id=point_id,
                        content=str(payload.get("content", "")),
                        source_document=str(payload.get("source_document", "")),
                        page_number=payload.get("page_number"),
                        chunk_index=payload.get("chunk_index"),
                        document_id=payload.get("document_id"),
                        chunk_id=payload.get("chunk_id") or point_id,
                        **extra,
                    ),
                )
            )
        return results

    async def delete(self, collection_name: str, vector_id: str) -> None:
        await self.delete_points(collection_name, [vector_id])

    async def delete_points(self, collection_name: str, vector_ids: list[str]) -> None:
        self._check()
        collection = self.collections.get(collection_name)
        if collection is None:
            return
        for point_id in vector_ids:
            self.deleted_points.append(point_id)
            collection["points"].pop(point_id, None)

    async def delete_collection(self, collection_name: str) -> None:
        self._check()
        self.collections.pop(collection_name, None)

    async def list_collections(self) -> list[dict[str, Any]]:
        self._check()
        return [{"name": name} for name in self.collections]

    async def close(self) -> None:
        return None


class FakeLLMClient:
    def __init__(self, answer: str = "The answer is supported by the document [1].") -> None:
        self.answer = answer
        self.unavailable = False
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        if self.unavailable:
            raise ModelUnavailableError()
        self.calls.append(
            {"model": model, "prompt": prompt, "system": system, "temperature": temperature}
        )
        return self.answer

    async def health(self, required_model: str | None = None) -> dict[str, Any]:
        if self.unavailable:
            return {"status": "unhealthy", "detail": "model service unavailable"}
        return {
            "status": "healthy",
            "model": required_model,
            "available_models": [required_model or "fake-llm"],
        }

    async def list_models(self) -> list[str]:
        return ["fake-llm"]

    async def close(self) -> None:
        return None


def make_container(
    settings: Settings,
    *,
    embedding: FakeEmbeddingProvider | None = None,
    vector_store: FakeVectorStore | None = None,
    llm: FakeLLMClient | None = None,
) -> AppContainer:
    audit = AuditStore(settings.database_url)
    registry = DocumentRegistry(settings.database_url)
    embedding = embedding or FakeEmbeddingProvider(model_name=settings.embedding_model)
    vector_store = vector_store or FakeVectorStore()
    llm = llm or FakeLLMClient()

    retrieval = RetrievalService(
        embedding_provider=embedding,
        vector_store=vector_store,
        collection_name=settings.qdrant_collection,
        default_top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    ingestion = IngestionService(
        settings=settings,
        registry=registry,
        embedding_provider=embedding,
        vector_store=vector_store,
        audit=audit,
    )
    query = QueryService(
        settings=settings,
        registry=registry,
        retrieval=retrieval,
        vector_store=vector_store,
        llm_client=llm,
        audit=audit,
    )
    return AppContainer(
        settings=settings,
        audit=audit,
        registry=registry,
        embedding_provider=embedding,
        vector_store=vector_store,
        llm_client=llm,
        ingestion_service=ingestion,
        retrieval_service=retrieval,
        query_service=query,
    )


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'sirus-test.db').as_posix()}",
        chunk_size=200,
        chunk_overlap=40,
        embedding_batch_size=4,
        retrieval_top_k=4,
        retrieval_score_threshold=0.0,
        ocr_enabled=False,
        max_page_count=20,
        max_chunks_per_document=500,
        request_timeout_seconds=1.0,
    )


@pytest.fixture
def fakes(test_settings: Settings) -> tuple[FakeEmbeddingProvider, FakeVectorStore, FakeLLMClient]:
    return (
        FakeEmbeddingProvider(model_name=test_settings.embedding_model),
        FakeVectorStore(),
        FakeLLMClient(),
    )


@pytest.fixture
def container(test_settings: Settings, fakes) -> AppContainer:
    embedding, vector_store, llm = fakes
    return make_container(
        test_settings, embedding=embedding, vector_store=vector_store, llm=llm
    )


@pytest.fixture
def client(test_settings: Settings, container: AppContainer):
    app = create_app(settings=test_settings, container=container)
    with TestClient(app) as test_client:
        yield test_client
