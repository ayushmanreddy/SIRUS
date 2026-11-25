from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.deps import get_container
from app.hardware import detect_hardware
from app.health import endpoint_host, readiness
from app.model_artifacts import verify_model_artifacts

router = APIRouter(tags=["operations"])


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "live"}


@router.get("/health/ready")
async def ready(request: Request) -> JSONResponse:
    container = get_container(request)
    checks = await readiness(container.settings, container)
    ready_now = all(check.status == "ready" for check in checks)
    payload = {
        "status": "ready" if ready_now else "degraded",
        "checks": [check.__dict__ for check in checks],
    }
    container.audit.record("service.readiness_checked", None, {"status": payload["status"]})
    return JSONResponse(status_code=200 if ready_now else 503, content=payload)


@router.get("/system/version")
async def version(request: Request) -> dict[str, str]:
    settings = get_container(request).settings
    return {"name": settings.app_name, "version": settings.app_version}


@router.get("/system/status")
async def system_status(request: Request) -> dict[str, object]:
    container = get_container(request)
    settings = container.settings
    checks = await readiness(settings, container)
    artifacts = verify_model_artifacts(settings)
    started_at = getattr(request.app.state, "started_at", time.monotonic())
    return {
        "profile": settings.app_profile,
        "uptime_seconds": round(time.monotonic() - started_at, 1),
        "models": {
            "answer_model": settings.answer_model,
            "embedding_model": settings.embedding_model,
        },
        "collection": settings.qdrant_collection,
        "service_hosts": {
            "llm": endpoint_host(settings.llm_service_url),
            "embeddings": endpoint_host(settings.embedding_service_url),
            "vector_store": endpoint_host(settings.vector_store_url),
        },
        "checks": [check.__dict__ for check in checks],
        "model_artifacts": artifacts.model_dump(),
    }


@router.get("/system/preflight")
async def preflight(request: Request) -> dict[str, object]:
    container = get_container(request)
    settings = container.settings
    artifacts = verify_model_artifacts(settings)
    return {
        "profile": settings.app_profile,
        "offline_endpoint_policy": (
            "enforced" if settings.app_profile == "offline-demo" else "not active"
        ),
        "hardware": detect_hardware().safe_dict(),
        "model_artifacts": artifacts.model_dump(),
    }
