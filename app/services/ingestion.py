from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.ai._embedding import EmbeddingProvider
from app.ai._vector_store import VectorStoreInterface
from app.audit import AuditStore
from app.config import Settings
from app.documents.extraction import ExtractedDocument, extract_document
from app.errors import (
    CollectionCompatibilityError,
    CollectionNotFoundError,
    DocumentTooLargeToIndexError,
    EmbeddingResponseError,
    NoExtractableTextError,
)
from app.rag.chunking import chunk_text
from app.services.document_registry import DocumentRecord, DocumentRegistry

logger = logging.getLogger("sirus.services.ingestion")

_POINT_NAMESPACE = uuid.UUID("4f6c6f83-5b1e-4f31-9f37-2a7f6c9c1d11")


@dataclass
class IngestionResult:
    record: DocumentRecord
    duplicate: bool


@dataclass
class _Chunk:
    page_number: int
    chunk_index: int
    text: str
    char_start: int | None
    char_end: int | None


class IngestionService:
    """Coordinates validation, extraction, chunking, embedding, and storage."""

    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreInterface,
        audit: AuditStore,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._embedding = embedding_provider
        self._vector_store = vector_store
        self._audit = audit

    async def ingest(self, filename: str | None, content_type: str | None, data: bytes) -> IngestionResult:
        extracted = await asyncio.to_thread(
            extract_document, filename, content_type, data, self._settings
        )
        model_name = await self._embedding.get_model_name()
        collection = self._settings.qdrant_collection

        duplicate = await asyncio.to_thread(
            self._registry.find_duplicate,
            extracted.sha256,
            model_name,
            collection,
        )
        if duplicate is not None:
            logger.info("Duplicate document upload detected: %s", duplicate.document_id)
            self._audit.record(
                "document.duplicate_skipped",
                None,
                {"document_id": duplicate.document_id, "chunk_count": duplicate.chunk_count},
            )
            return IngestionResult(record=duplicate, duplicate=True)

        chunks = self._build_chunks(extracted)
        if not chunks:
            raise NoExtractableTextError()
        if len(chunks) > self._settings.max_chunks_per_document:
            raise DocumentTooLargeToIndexError(
                f"The document produced {len(chunks)} chunks, exceeding the configured "
                f"limit of {self._settings.max_chunks_per_document}. Split the document and retry."
            )

        dimension = await self._ensure_collection(model_name)
        vectors = await self._embed_chunks([chunk.text for chunk in chunks])
        for vector in vectors:
            if len(vector) != dimension:
                raise CollectionCompatibilityError(
                    f"The embedding model returned dimension {len(vector)}, but collection "
                    f"'{collection}' expects {dimension}. Rebuild the collection after changing "
                    "embedding models."
                )

        document_id = str(uuid.uuid4())
        point_ids: list[str] = []
        points: list[dict[str, Any]] = []
        chunk_summaries: list[dict[str, Any]] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = str(
                uuid.uuid5(
                    _POINT_NAMESPACE,
                    f"{document_id}:{chunk.page_number}:{chunk.chunk_index}",
                )
            )
            point_ids.append(point_id)
            payload = {
                "id": point_id,
                "chunk_id": point_id,
                "document_id": document_id,
                "source_document": extracted.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "embedding_model": model_name,
                "sha256": extracted.sha256,
            }
            points.append({"id": point_id, "vector": vector, "payload": payload})
            chunk_summaries.append(
                {
                    "chunk_id": point_id,
                    "page": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text_preview": chunk.text[:160],
                }
            )

        try:
            await self._vector_store.upsert_points(collection, points)
        except Exception:
            logger.exception("Vector upsert failed for document %s; cleaning up partial points.", document_id)
            await self._cleanup_points(collection, point_ids)
            raise

        record = DocumentRecord(
            document_id=document_id,
            filename=extracted.filename,
            content_type=extracted.content_type,
            sha256=extracted.sha256,
            size_bytes=extracted.size_bytes,
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            embedding_model=model_name,
            embedding_dimension=dimension,
            collection_name=collection,
            created_at=datetime.now(UTC).isoformat(),
            chunks=chunk_summaries,
        )

        try:
            await asyncio.to_thread(self._registry.add, record)
        except sqlite3.IntegrityError:
            # Concurrent duplicate upload: keep the existing document, discard ours.
            await self._cleanup_points(collection, point_ids)
            existing = await asyncio.to_thread(
                self._registry.find_duplicate,
                extracted.sha256,
                model_name,
                collection,
            )
            if existing is not None:
                return IngestionResult(record=existing, duplicate=True)
            raise
        except Exception:
            await self._cleanup_points(collection, point_ids)
            raise

        self._audit.record(
            "document.ingested",
            None,
            {
                "document_id": document_id,
                "file_type": extracted.file_type,
                "page_count": extracted.page_count,
                "chunk_count": len(chunks),
                "embedding_model": model_name,
            },
        )
        logger.info(
            "Ingested document %s (%s chunks, %s pages, %s)",
            document_id,
            len(chunks),
            extracted.page_count,
            model_name,
        )
        return IngestionResult(record=record, duplicate=False)

    def _build_chunks(self, extracted: ExtractedDocument) -> list[_Chunk]:
        chunks: list[_Chunk] = []
        for page in extracted.pages:
            if not page.text.strip():
                continue
            for text, metadata in chunk_text(
                page.text,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
                page_number=page.page_number,
            ):
                chunks.append(
                    _Chunk(
                        page_number=metadata.page_number,
                        chunk_index=metadata.chunk_index,
                        text=text,
                        char_start=metadata.char_start,
                        char_end=metadata.char_end,
                    )
                )
        return chunks

    async def _ensure_collection(self, model_name: str) -> int:
        collection = self._settings.qdrant_collection
        dimension = await self._embedding.get_dimension()

        try:
            info = await self._vector_store.get_collection_info(collection)
        except CollectionNotFoundError:
            info = None

        if info is None:
            await self._vector_store.create_collection(collection, dimension)
            await asyncio.to_thread(
                self._registry.set_collection_state, collection, model_name, dimension
            )
            return dimension

        from app.ai.vector_store_qdrant import QdrantClient

        existing_dimension = QdrantClient.vector_size_from_info(info)
        if existing_dimension is not None and existing_dimension != dimension:
            raise CollectionCompatibilityError(
                f"Collection '{collection}' stores {existing_dimension}-dimensional vectors, "
                f"but embedding model '{model_name}' returns {dimension}. Create a new "
                "collection (set QDRANT_COLLECTION) or rebuild the collection."
            )

        state = await asyncio.to_thread(self._registry.get_collection_state, collection)
        if state is not None and (
            state.embedding_model != model_name or state.embedding_dimension != dimension
        ):
            raise CollectionCompatibilityError(
                f"Collection '{collection}' was built with embedding model '{state.embedding_model}' "
                f"({state.embedding_dimension} dimensions), but the configured model is "
                f"'{model_name}' ({dimension} dimensions). Rebuild the collection before ingesting."
            )
        if state is None:
            # A pre-existing collection (for example created before the registry
            # existed) is adopted only when its dimension matches the live model.
            await asyncio.to_thread(
                self._registry.set_collection_state, collection, model_name, dimension
            )
        return dimension

    async def _embed_chunks(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_vectors = await self._embedding.embed_batch(batch)
            if len(batch_vectors) != len(batch):
                raise EmbeddingResponseError(
                    f"Embedding count mismatch: expected {len(batch)}, received {len(batch_vectors)}."
                )
            vectors.extend(batch_vectors)
        return vectors

    async def _cleanup_points(self, collection: str, point_ids: list[str]) -> None:
        try:
            await self._vector_store.delete_points(collection, point_ids)
        except Exception as error:  # noqa: BLE001 - best-effort cleanup
            logger.warning("Could not clean up partial points in %s: %s", collection, error)
