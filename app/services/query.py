from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.ai._vector_store import VectorStoreInterface
from app.audit import AuditStore
from app.config import Settings
from app.errors import CollectionNotFoundError, DocumentNotFoundError, VectorStoreUnavailableError
from app.llm_client import ModelUnavailableError, OllamaClient
from app.rag.prompts import NO_EVIDENCE_ANSWER, build_rag_prompt, is_insufficient_answer
from app.rag.retrieval import RetrievalService, RetrievedChunk
from app.services.document_registry import DocumentRegistry

logger = logging.getLogger("sirus.services.query")


@dataclass
class QueryResult:
    answer: str
    sources: list[RetrievedChunk]
    model: str
    grounded: bool


class QueryService:
    """Retrieval + grounded generation orchestration."""

    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        retrieval: RetrievalService,
        vector_store: VectorStoreInterface,
        llm_client: OllamaClient,
        audit: AuditStore,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._retrieval = retrieval
        self._vector_store = vector_store
        self._llm = llm_client
        self._audit = audit

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> QueryResult:
        question = question.strip()
        model = self._settings.answer_model

        if document_id:
            record = await asyncio.to_thread(self._registry.get, document_id)
            registry_count = await asyncio.to_thread(self._registry.count)
            # An empty registry may indicate legacy vectors created before the
            # registry existed; allow filtering those by their Qdrant payload.
            if record is None and registry_count > 0:
                raise DocumentNotFoundError(f"Document '{document_id}' is not indexed.")

        if not await self._has_indexed_documents():
            return QueryResult(
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                model=model,
                grounded=False,
            )

        sources = await self._retrieval.retrieve(
            question,
            top_k=top_k,
            document_id=document_id,
        )
        if not sources:
            return QueryResult(
                answer=NO_EVIDENCE_ANSWER,
                sources=[],
                model=model,
                grounded=False,
            )

        system_prompt, user_prompt = build_rag_prompt(question, sources)
        try:
            generated = await self._llm.generate(
                model,
                user_prompt,
                system=system_prompt,
                temperature=self._settings.llm_temperature,
            )
        except ModelUnavailableError:
            logger.error("Answer generation failed because the LLM service is unavailable.")
            raise

        answer = generated.strip()
        if answer.upper().startswith("ANSWER:"):
            answer = answer[len("ANSWER:") :].strip()

        grounded = bool(answer) and not is_insufficient_answer(answer)
        if not grounded:
            answer = NO_EVIDENCE_ANSWER

        self._audit.record(
            "document.question_answered",
            None,
            {
                "source_count": len(sources),
                "grounded": grounded,
                "model": model,
            },
        )
        return QueryResult(answer=answer, sources=sources, model=model, grounded=grounded)

    async def _has_indexed_documents(self) -> bool:
        if await asyncio.to_thread(self._registry.count) > 0:
            return True
        try:
            info = await self._vector_store.get_collection_info(self._settings.qdrant_collection)
        except CollectionNotFoundError:
            return False
        except Exception as error:  # noqa: BLE001 - normalized to a safe dependency error
            logger.error("Could not inspect the vector collection: %s", error)
            raise VectorStoreUnavailableError() from error

        from app.ai.vector_store_qdrant import QdrantClient

        point_count = QdrantClient.point_count_from_info(info)
        return point_count is None or point_count > 0
