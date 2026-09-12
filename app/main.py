from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.ocr import InvalidPDFError

from app.llm_client import OllamaClient, ModelUnavailableError
from app.audit import AuditStore, configure_logging, log_event
from app.config import Settings
from app.ocr import render_pdf_pages, extract_text
from app.hardware import detect_hardware
from app.health import endpoint_host, readiness
from app.model_artifacts import verify_model_artifacts
from app.router import TaskType, classify_task, pick_model

settings = Settings()
audit = AuditStore(settings.database_url)
logger = configure_logging(settings.log_level)
llm_client = OllamaClient(
    settings.llm_service_url,
    settings.generation_timeout_seconds,
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


@app.post("/documents/analyze", tags=["documents"])
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Analyze an uploaded document by extracting text and generating a summary.

    Accepts a PDF file, extracts text via OCR, and generates a summary using
    the local reasoning model.
    """
    import io

    # Read the uploaded file bytes
    file_bytes = await file.read()

    # Extract text from the PDF
    try:
        extracted_text = extract_text(file_bytes)
    except InvalidPDFError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # Build the summarization prompt
    prompt = f"Summarize the key findings from this inspection report:\n\n{extracted_text}"

    # Generate summary using the local reasoning model
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