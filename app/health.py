from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.config import Settings


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


async def check_http_service(name: str, endpoint: str, timeout: float) -> CheckResult:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(endpoint)
        if response.status_code < 500:
            return CheckResult(name, "ready", "reachable")
        return CheckResult(name, "unavailable", "service returned an error")
    except (httpx.HTTPError, ValueError):
        return CheckResult(name, "unavailable", "not reachable")


def check_database(settings: Settings) -> CheckResult:
    try:
        path = Path(settings.database_url.removeprefix("sqlite:///"))
        with sqlite3.connect(path) as connection:
            connection.execute("SELECT 1 FROM audit_events LIMIT 1")
        return CheckResult("database", "ready", "migration available")
    except sqlite3.Error:
        return CheckResult("database", "unavailable", "migration missing or database unavailable")


async def readiness(settings: Settings) -> list[CheckResult]:
    vector_health = f"{settings.vector_store_url}/healthz"
    # A generic root reachability check permits different local LLM/embedding serving runtimes.
    checks = await asyncio.gather(
        check_http_service("llm", settings.llm_service_url, settings.request_timeout_seconds),
        check_http_service("embeddings", settings.embedding_service_url, settings.request_timeout_seconds),
        check_http_service("vector_store", vector_health, settings.request_timeout_seconds),
    )
    return [check_database(settings), *checks]


def endpoint_host(endpoint: str) -> str:
    return urlparse(endpoint).hostname or "unknown"
