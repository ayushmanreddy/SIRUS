from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.config import Settings
from app.dependencies import AppContainer

logger = logging.getLogger("sirus.health")


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _database_check(settings: Settings) -> CheckResult:
    try:
        path = Path(settings.database_path)
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('audit_events','documents')"
            ).fetchall()
        present = {row[0] for row in rows}
        if {"audit_events", "documents"} <= present:
            return CheckResult("database", "ready", "schema available")
        return CheckResult("database", "unavailable", "database schema is not initialized")
    except (sqlite3.Error, ValueError):
        return CheckResult("database", "unavailable", "database unavailable")


async def readiness(settings: Settings, container: AppContainer) -> list[CheckResult]:
    """Check process dependencies, including whether required models are installed.

    A reachable Ollama process is not the same as a usable model. Embedding and
    LLM health checks inspect ``/api/tags`` so readiness cannot claim success
    when the configured model has not been pulled.
    """
    database = _database_check(settings)

    vector = await container.vector_store.health()
    vector_ready = vector.get("status") == "healthy"
    vector_check = CheckResult(
        "vector_store",
        "ready" if vector_ready else "unavailable",
        "reachable" if vector_ready else str(vector.get("detail", "unavailable")),
    )

    embedding = await container.embedding_provider.health()
    embedding_ready = embedding.get("status") == "healthy"
    embedding_check = CheckResult(
        "embeddings",
        "ready" if embedding_ready else "unavailable",
        (
            f"model '{settings.embedding_model}' available"
            if embedding_ready
            else str(embedding.get("detail", "embedding service unavailable"))
        ),
    )

    llm = await container.llm_client.health(required_model=settings.answer_model)
    llm_ready = llm.get("status") == "healthy"
    llm_check = CheckResult(
        "llm",
        "ready" if llm_ready else "unavailable",
        (
            f"model '{settings.answer_model}' available"
            if llm_ready
            else str(llm.get("detail", "model service unavailable"))
        ),
    )

    return [database, vector_check, embedding_check, llm_check]


def endpoint_host(endpoint: str) -> str:
    return urlparse(endpoint).hostname or "unknown"
