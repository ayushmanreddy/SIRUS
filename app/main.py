from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.ocr import InvalidPDFError, render_pdf_pages
from app.llm_client import OllamaClient, ModelUnavailableError
from app.audit import AuditStore, configure_logging, log_event
from app.sandbox import run_python
from app.config import Settings
from app.ocr import extract_text
from app.hardware import detect_hardware
from app.health import endpoint_host, readiness
from app.model_artifacts import verify_model_artifacts
from app.router import TaskType, classify_task, pick_model
from app.rag.chunking import chunk_text
from app.ai.embedding_ollama import OllamaEmbeddingProvider

settings = Settings()
audit = AuditStore(settings.database_url)
logger = configure_logging(settings.log_level)
llm_client = OllamaClient(
    settings.llm_service_url,
    settings.generation_timeout_seconds,
)
embedding_provider = OllamaEmbeddingProvider(
    base_url=settings.embedding_service_url,
    model_name=settings.embedding_model,
)
started_at = time.monotonic()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    audit.initialize()
    audit.record("service.started", None, {"profile": settings.app_profile, "version": settings.app_version})
    log_event(logger, "service.started", profile=settings.app_profile, version=settings.app_version)
    yield
    log_event(logger, "service.stopped")


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.get("/health/live", tags=["operations"])
async def liveness() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready", tags=["operations"])
async def ready() -> JSONResponse:
    checks = await readiness(settings)
    ready_now = all(check.status == "ready" for check in checks)
    payload = {"status": "ready" if ready_now else "degraded", "checks": [check.__dict__ for check in checks]}
    audit.record("service.readiness_checked", None, {"status": payload["status"]})
    return JSONResponse(status_code=200 if ready_now else 503, content=payload)


@app.get("/system/version", tags=["operations"])
async def version() -> dict[str, str]:
    return {"name": settings.app_name, "version": settings.app_version}


@app.get("/system/status", tags=["operations"])
async def system_status() -> dict[str, object]:
    checks = await readiness(settings)
    artifacts = verify_model_artifacts(settings)
    return {
        "profile": settings.app_profile,
        "uptime_seconds": round(time.monotonic() - started_at, 1),
        "service_hosts": {
            "llm": endpoint_host(settings.llm_service_url),
            "embeddings": endpoint_host(settings.embedding_service_url),
            "vector_store": endpoint_host(settings.vector_store_url),
        },
        "checks": [check.__dict__ for check in checks],
        "model_artifacts": artifacts.model_dump(),
    }


@app.get("/system/preflight", tags=["operations"])
async def preflight() -> dict[str, object]:
    """Local-demo evidence without exposing model paths, secrets, or host identifiers."""
    artifacts = verify_model_artifacts(settings)
    return {
        "profile": settings.app_profile,
        "offline_endpoint_policy": "enforced" if settings.app_profile == "offline-demo" else "not active",
        "hardware": detect_hardware().safe_dict(),
        "model_artifacts": artifacts.model_dump(),
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    task_type: TaskType | None = None
    has_image: bool = False


class CodeExecutionRequest(BaseModel):
    code: str = Field(max_length=5000)


class CodeExecutionResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool


class GenerateResponse(BaseModel):
    task_type: str
    model: str
    response: str


@app.post("/agent/generate", response_model=GenerateResponse, tags=["agent"])
async def generate_agent_response(
    payload: GenerateRequest,
    request: Request,
) -> GenerateResponse:
    task = payload.task_type or classify_task(
        payload.prompt,
        has_image=payload.has_image,
    )
    model = pick_model(settings, task)
    correlation_id = request.state.correlation_id

    try:
        generated_text = await llm_client.generate(model, payload.prompt)

        audit.record(
            "agent.generation_succeeded",
            correlation_id,
            {"task_type": task.value, "model": model},
        )

        return GenerateResponse(task_type=task.value, model=model, response=generated_text)

    except ModelUnavailableError as error:
        audit.record(
            "agent.generation_failed",
            correlation_id,
            {"task_type": task.value, "model": model, "reason": "model_unavailable"},
        )

        raise HTTPException(status_code=502, detail="Local model service is unavailable.") from error


@app.post("/agent/execute_code", response_model=CodeExecutionResponse, tags=["agent"])
async def execute_code(request: Request, payload: CodeExecutionRequest) -> CodeExecutionResponse:
    correlation_id = request.state.correlation_id
    result = run_python(payload.code)
    audit.record(
        "code_execution",
        correlation_id,
        {"task_type": "code_execution", "exit_code": result["exit_code"], "timed_out": result["timed_out"]},
    )
    return CodeExecutionResponse(
        stdout=result["stdout"],
        stderr=result["stderr"],
        exit_code=result["exit_code"],
        timed_out=result["timed_out"],
    )


class IngestResponse(BaseModel):
    document_id: str
    chunks: list[dict]


class DocumentQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)
    document_id: str | None = None


class SourceInfo(BaseModel):
    document_id: str
    page: int
    chunk_index: int
    score: float
    text_preview: str


class DocumentQueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


@app.post("/documents/ingest", response_model=IngestResponse, tags=["documents"])
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
) -> IngestResponse:
    """Ingest a PDF document: extract text, chunk, embed, and store in Qdrant."""
    import io

    # Read the uploaded file bytes
    file_bytes = await file.read()

    # Enforce max upload size
    max_size = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum {settings.max_upload_mb}MB.")

    # Check page count
    import fitz

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as error:
        raise HTTPException(status_code=422, detail="Failed to open PDF") from error

    page_count = len(doc)
    if page_count > settings.max_page_count:
        raise HTTPException(
            status_code=422,
            detail=f"Too many pages. Maximum {settings.max_page_count} pages.",
        )

    # Extract text per page using OCR
    pages_png = render_pdf_pages(file_bytes)

    from app.ocr import _get_reader
    reader = _get_reader()

    all_text: list[str] = []
    for page_png in pages_png:
        import cv2
        import numpy as np
        image_np = np.frombuffer(page_png, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        results = reader.readtext(image_bgr, detail=0)
        page_text = "\n".join(results)
        all_text.append(page_text)

    doc.close()

    # Concatenate all page text
    extracted_text = "\n\n".join(all_text)

    # Reject empty extracted text
    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the document.")

    # Chunk the text, preserving page numbers
    chunks_data: list[tuple[str, int, int]] = []  # (chunk_text, page_number, chunk_index)
    for page_num, page_text in enumerate(all_text, start=1):
        if not page_text.strip():
            continue
        chunk_results = chunk_text(page_text, chunk_size=512, chunk_overlap=64, page_number=page_num)
        for chunk_idx, (chunk_text, metadata) in enumerate(chunk_results):
            chunks_data.append((chunk_text, metadata.page_number, metadata.chunk_index))

    # Generate embeddings and store in Qdrant "documents" collection
    document_id = str(uuid.uuid4())
    collection_name = "documents"

    # Ensure collection exists
    from app.ai.vector_store_qdrant import QdrantClient

    qdrant = QdrantClient(base_url=settings.vector_store_url)
    try:
        collection_info = await qdrant.get_collection_info(collection_name)
        if not collection_info:
            await qdrant.create_collection(
                collection_name=collection_name,
                dimension=await embedding_provider.get_dimension(),
            )
    except Exception:
        await qdrant.create_collection(
            collection_name=collection_name,
            dimension=await embedding_provider.get_dimension(),
        )

    # Insert chunks with embeddings
    stored_chunks: list[dict] = []
    for chunk_idx, (chunk_text, page_number, chunk_index) in enumerate(chunks_data):
        embedding = await embedding_provider.embed(chunk_text)
        metadata = type("Meta", (), {"page_number": page_number, "chunk_index": chunk_index, "content": chunk_text})()
        await qdrant.insert(
            collection_name=collection_name,
            vector=embedding,
            metadata=metadata,
        )
        stored_chunks.append({
            "chunk_text": chunk_text,
            "page": page_number,
            "chunk_index": chunk_index,
        })

    return IngestResponse(document_id=document_id, chunks=stored_chunks)


@app.post("/documents/query", response_model=DocumentQueryResponse, tags=["documents"])
async def query_documents(
    payload: DocumentQueryRequest,
) -> DocumentQueryResponse:
    """Query documents: embed question, similarity search, generate answer."""
    from app.ai.vector_store_qdrant import QdrantClient

    qdrant = QdrantClient(base_url=settings.vector_store_url)

    # Embed the question
    question_embedding = await embedding_provider.embed(payload.question)

    # Determine collection name
    collection_name = payload.document_id if payload.document_id else "documents"

    # Similarity search
    filter_metadata = {}
    if payload.document_id:
        filter_metadata["document_id"] = payload.document_id

    results: list = await qdrant.similarity_search(
        collection_name=collection_name,
        query_vector=question_embedding,
        top_k=payload.top_k,
        filter_metadata=filter_metadata,
    )

    if not results:
        # Answer "not found in the document"
        return DocumentQueryResponse(
            answer="not found in the document",
            sources=[],
        )

    # Build context from results
    context_parts: list[str] = []
    sources: list[dict] = []

    for result in results:
        meta = result.metadata
        text_preview = (meta.content[:100] + "...") if len(meta.content) > 100 else meta.content
        context_parts.append(f"[Page {meta.page_number}, Chunk {meta.chunk_index}] {meta.content}")
        sources.append({
            "document_id": meta.source_document or collection_name,
            "page": meta.page_number,
            "chunk_index": meta.chunk_index,
            "score": round(result.score, 4),
            "text_preview": text_preview,
        })

    context = "\n\n".join(context_parts)

    # Build prompt instructing model to answer ONLY from provided context
    prompt = f"""Answer the user's question based ONLY on the provided context below. If the answer is not found in the context, say "not found in the document".

Context:
{context}

Question: {payload.question}

Answer:"""

    # Generate response using the local reasoning model
    try:
        generated_text = await llm_client.generate(settings.reasoning_model, prompt)
    except ModelUnavailableError:
        generated_text = "Local model service is unavailable."

    # Trim the "Answer:" prefix if present
    answer = generated_text.strip()
    if answer.upper().startswith("ANSWER:"):
        answer = answer[len("ANSWER:"):].strip()
    if not answer.strip():
        answer = "not found in the document"

    return DocumentQueryResponse(
        answer=answer,
        sources=sources,
    )

@app.post("/documents/analyze", tags=["documents"])
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Analyze an uploaded document by extracting text and generating a summary.

    Accepts a PDF file, extracts text via OCR, and generates a summary using
    the local reasoning model.
    """
    file_bytes = await file.read()

    try:
        extracted_text = extract_text(file_bytes)
    except InvalidPDFError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    prompt = f"Summarize the key findings from this inspection report:\n\n{extracted_text}"

    try:
        summary = await llm_client.generate(settings.reasoning_model, prompt)

        audit.record(
            "document.analyzed",
            request.state.correlation_id,
            {"task_type": "summarization", "model": settings.reasoning_model},
        )

        return {"extracted_text": extracted_text, "summary": summary, "model": settings.reasoning_model}

    except ModelUnavailableError as error:
        audit.record(
            "document.analysis_failed",
            request.state.correlation_id,
            {"task_type": "summarization", "model": settings.reasoning_model, "reason": "model_unavailable"},
        )

        raise HTTPException(status_code=502, detail="Local model service is unavailable.") from error