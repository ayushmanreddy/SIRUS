from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.audit import AuditStore, configure_logging, log_event
from app.config import Settings
from app.health import endpoint_host, readiness

settings = Settings()
audit = AuditStore(settings.database_url)
logger = configure_logging(settings.log_level)
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
    try:
        response = await call_next(request)
    except Exception:
        audit.record("api.error", correlation_id, {"path": request.url.path, "method": request.method})
        log_event(logger, "api.error", correlation_id=correlation_id, path=request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": correlation_id})
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
    return {
        "profile": settings.app_profile,
        "uptime_seconds": round(time.monotonic() - started_at, 1),
        "service_hosts": {
            "llm": endpoint_host(settings.llm_service_url),
            "embeddings": endpoint_host(settings.embedding_service_url),
            "vector_store": endpoint_host(settings.vector_store_url),
        },
        "checks": [check.__dict__ for check in checks],
    }
