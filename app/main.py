from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import agent, documents, system
from app.audit import configure_logging, log_event
from app.config import Settings
from app.dependencies import AppContainer, build_container
from app.errors import SirusError

logger = logging.getLogger("sirus.main")


def create_app(
    settings: Settings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    """Application factory.

    Passing a container allows tests to substitute fake embedding, vector-store,
    and LLM adapters without touching route code.
    """
    settings = settings or (container.settings if container else Settings())
    container = container or build_container(settings)
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        container.audit.initialize()
        container.registry.initialize()
        container.audit.record(
            "service.started",
            None,
            {"profile": settings.app_profile, "version": settings.app_version},
        )
        log_event(
            logger,
            "service.started",
            profile=settings.app_profile,
            version=settings.app_version,
        )
        try:
            yield
        finally:
            await container.aclose()
            log_event(logger, "service.stopped")

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.state.container = container
    app.state.started_at = time.monotonic()

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        return response

    @app.exception_handler(SirusError)
    async def sirus_error_handler(request: Request, error: SirusError) -> JSONResponse:
        logger.warning(
            "Request failed: %s %s -> %s (%s)",
            request.method,
            request.url.path,
            error.status_code,
            type(error).__name__,
        )
        return JSONResponse(
            status_code=error.status_code,
            content={
                "detail": error.detail,
                "request_id": getattr(request.state, "correlation_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, error: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error for %s %s: %s",
            request.method,
            request.url.path,
            type(error).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "request_id": getattr(request.state, "correlation_id", None),
            },
        )

    app.include_router(system.router)
    app.include_router(documents.router)
    app.include_router(agent.router)
    return app


app = create_app()
