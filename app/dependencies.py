from __future__ import annotations

from dataclasses import dataclass

from app.ai._embedding import EmbeddingProvider
from app.ai._vector_store import VectorStoreInterface
from app.ai.embedding_ollama import OllamaEmbeddingProvider
from app.ai.vector_store_qdrant import QdrantClient
from app.audit import AuditStore
from app.config import Settings
from app.llm_client import OllamaClient
from app.rag.retrieval import RetrievalService
from app.services.document_registry import DocumentRegistry
from app.services.ingestion import IngestionService
from app.services.query import QueryService


@dataclass
class AppContainer:
    """Application-scoped dependencies, assembled once at startup."""

    settings: Settings
    audit: AuditStore
    registry: DocumentRegistry
    embedding_provider: EmbeddingProvider
    vector_store: VectorStoreInterface
    llm_client: OllamaClient
    ingestion_service: IngestionService
    retrieval_service: RetrievalService
    query_service: QueryService

    async def aclose(self) -> None:
        await self.embedding_provider.close()
        await self.vector_store.close()
        await self.llm_client.close()


def build_container(settings: Settings) -> AppContainer:
    audit = AuditStore(settings.database_url)
    registry = DocumentRegistry(settings.database_url)
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.embedding_service_url,
        model_name=settings.embedding_model,
    )
    vector_store = QdrantClient(
        base_url=settings.vector_store_url,
        timeout=max(settings.request_timeout_seconds, settings.vector_store_timeout_seconds),
    )
    llm_client = OllamaClient(
        settings.llm_service_url,
        settings.generation_timeout_seconds,
    )
    retrieval = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        collection_name=settings.qdrant_collection,
        default_top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    ingestion = IngestionService(
        settings=settings,
        registry=registry,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        audit=audit,
    )
    query = QueryService(
        settings=settings,
        registry=registry,
        retrieval=retrieval,
        vector_store=vector_store,
        llm_client=llm_client,
        audit=audit,
    )
    return AppContainer(
        settings=settings,
        audit=audit,
        registry=registry,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_client=llm_client,
        ingestion_service=ingestion,
        retrieval_service=retrieval,
        query_service=query,
    )
