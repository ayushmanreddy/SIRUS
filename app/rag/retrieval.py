from __future__ import annotations

import logging
from dataclasses import dataclass

from app.ai._embedding import EmbeddingProvider
from app.ai._vector_store import VectorStoreInterface
from app.errors import CollectionNotFoundError

logger = logging.getLogger("sirus.rag.retrieval")


@dataclass(frozen=True)
class RetrievedChunk:
    citation_id: int
    document_id: str
    document_name: str
    chunk_id: str
    page_number: int | None
    chunk_index: int | None
    content: str
    score: float


class RetrievalService:
    """Embed a question, search Qdrant, and return traceable chunks.

    This class depends only on the embedding and vector-store abstractions, so
    retrieval can be tested without an LLM or HTTP routes.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreInterface,
        collection_name: str,
        default_top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> None:
        self._embedding = embedding_provider
        self._vector_store = vector_store
        self._collection = collection_name
        self._default_top_k = default_top_k
        self._score_threshold = score_threshold

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        limit = top_k or self._default_top_k
        threshold = self._score_threshold if score_threshold is None else score_threshold

        query_vector = await self._embedding.embed(question.strip())
        filter_metadata = {"document_id": document_id} if document_id else None

        try:
            results = await self._vector_store.similarity_search(
                collection_name=self._collection,
                query_vector=query_vector,
                top_k=limit,
                filter_metadata=filter_metadata,
            )
        except CollectionNotFoundError:
            logger.info("Vector collection '%s' does not exist yet; no evidence retrieved.", self._collection)
            return []

        chunks: list[RetrievedChunk] = []
        for result in results:
            if result.score < threshold:
                continue
            metadata = result.metadata
            if not metadata.content.strip():
                continue
            chunks.append(
                RetrievedChunk(
                    citation_id=len(chunks) + 1,
                    document_id=metadata.document_id or "",
                    document_name=metadata.source_document or "unknown",
                    chunk_id=metadata.chunk_id or result.vector_id,
                    page_number=metadata.page_number,
                    chunk_index=metadata.chunk_index,
                    content=metadata.content,
                    score=round(float(result.score), 6),
                )
            )

        chunks.sort(key=lambda item: item.score, reverse=True)
        # Re-number after sorting so citation [n] matches response order.
        chunks = [
            RetrievedChunk(
                citation_id=index,
                document_id=item.document_id,
                document_name=item.document_name,
                chunk_id=item.chunk_id,
                page_number=item.page_number,
                chunk_index=item.chunk_index,
                content=item.content,
                score=item.score,
            )
            for index, item in enumerate(chunks, start=1)
        ]
        logger.info(
            "Retrieved %s chunk(s) above threshold %.3f for query (document filter=%s)",
            len(chunks),
            threshold,
            document_id or "none",
        )
        return chunks
