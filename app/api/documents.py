from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_container
from app.dependencies import AppContainer
from app.documents.extraction import extract_document
from app.errors import FileTooLargeError, InvalidDocumentError
from app.llm_client import ModelUnavailableError

logger = logging.getLogger("sirus.api.documents")
router = APIRouter(tags=["documents"])

_READ_CHUNK_BYTES = 1024 * 1024


class ChunkSummary(BaseModel):
    chunk_id: str
    page: int | None
    chunk_index: int | None
    text_preview: str


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    duplicate: bool
    embedding_model: str
    chunks: list[ChunkSummary]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    content_type: str
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    created_at: str


class DocumentDetail(DocumentSummary):
    size_bytes: int
    collection_name: str
    chunks: list[ChunkSummary]


class DocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    documents: list[DocumentSummary]


class DocumentQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_id: str | None = Field(default=None, max_length=128)

    @field_validator("question")
    @classmethod
    def question_must_have_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class SourceReference(BaseModel):
    citation_id: int
    document_id: str
    document_name: str
    chunk_id: str
    page: int | None
    chunk_index: int | None
    score: float
    text_preview: str


class DocumentQueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    model: str
    grounded: bool


class AnalyzeResponse(BaseModel):
    filename: str
    extracted_text: str
    summary: str
    model: str
    page_count: int
    truncated: bool


async def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    if not upload.filename:
        raise InvalidDocumentError("The upload must include a filename.")
    buffer = bytearray()
    while True:
        chunk = await upload.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        if len(buffer) + len(chunk) > max_bytes:
            raise FileTooLargeError(
                f"File too large. Maximum size is {max_bytes // (1024 * 1024)} MB."
            )
        buffer.extend(chunk)
    return bytes(buffer)


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> IngestResponse:
    container: AppContainer = get_container(request)
    max_bytes = container.settings.max_upload_mb * 1024 * 1024
    data = await _read_upload_limited(file, max_bytes)
    result = await container.ingestion_service.ingest(file.filename, file.content_type, data)
    record = result.record
    return IngestResponse(
        document_id=record.document_id,
        filename=record.filename,
        page_count=record.page_count,
        chunk_count=record.chunk_count,
        duplicate=result.duplicate,
        embedding_model=record.embedding_model,
        chunks=[ChunkSummary(**chunk) for chunk in record.chunks],
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    container: AppContainer = get_container(request)
    records, total = await asyncio.to_thread(container.registry.list_documents, limit, offset)
    return DocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        documents=[DocumentSummary(**record.__dict__) for record in records],
    )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(request: Request, document_id: str) -> DocumentDetail:
    container: AppContainer = get_container(request)
    record = await asyncio.to_thread(container.registry.get, document_id)
    if record is None:
        from app.errors import DocumentNotFoundError

        raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
    return DocumentDetail(
        **DocumentSummary(**record.__dict__).model_dump(),
        size_bytes=record.size_bytes,
        collection_name=record.collection_name,
        chunks=[ChunkSummary(**chunk) for chunk in record.chunks],
    )


@router.post("/documents/query", response_model=DocumentQueryResponse)
async def query_documents(
    request: Request,
    payload: DocumentQueryRequest,
) -> DocumentQueryResponse:
    container: AppContainer = get_container(request)
    result = await container.query_service.answer(
        payload.question,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )
    sources = [
        SourceReference(
            citation_id=source.citation_id,
            document_id=source.document_id,
            document_name=source.document_name,
            chunk_id=source.chunk_id,
            page=source.page_number,
            chunk_index=source.chunk_index,
            score=source.score,
            text_preview=(
                source.content
                if len(source.content) <= 300
                else source.content[:300] + "..."
            ),
        )
        for source in result.sources
    ]
    return DocumentQueryResponse(
        answer=result.answer,
        sources=sources,
        model=result.model,
        grounded=result.grounded,
    )


@router.post("/documents/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> AnalyzeResponse:
    container: AppContainer = get_container(request)
    max_bytes = container.settings.max_upload_mb * 1024 * 1024
    data = await _read_upload_limited(file, max_bytes)
    extracted = await asyncio.to_thread(
        extract_document, file.filename, file.content_type, data, container.settings
    )
    full_text = extracted.text
    prompt_text = full_text[:12000]
    prompt = f"Summarize the key findings in this document:\n\n{prompt_text}"
    try:
        summary = await container.llm_client.generate(
            container.settings.answer_model,
            prompt,
            temperature=container.settings.llm_temperature,
        )
    except ModelUnavailableError:
        container.audit.record(
            "document.analysis_failed",
            request.state.correlation_id,
            {"task_type": "summarization", "reason": "model_unavailable"},
        )
        raise
    container.audit.record(
        "document.analyzed",
        request.state.correlation_id,
        {"task_type": "summarization", "page_count": extracted.page_count},
    )
    return AnalyzeResponse(
        filename=extracted.filename,
        extracted_text=full_text[:50000],
        summary=summary,
        model=container.settings.answer_model,
        page_count=extracted.page_count,
        truncated=len(full_text) > 50000,
    )
