from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {"authorization", "token", "secret", "password", "api_key", "prompt", "document", "content"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", value)
    return value


class AuditStore:
    def __init__(self, database_url: str) -> None:
        if not database_url.startswith("sqlite:///"):
            raise ValueError("Foundation MVP supports a local SQLite database only")
        self.path = Path(database_url.removeprefix("sqlite:///"))

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    correlation_id TEXT,
                    payload_json TEXT NOT NULL
                )"""
            )

    def record(self, event_name: str, correlation_id: str | None, payload: dict[str, Any]) -> None:
        safe_payload = json.dumps(redact(payload), sort_keys=True, default=str)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO audit_events (occurred_at, event_name, correlation_id, payload_json) VALUES (?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), event_name, correlation_id, safe_payload),
            )


def configure_logging(level: str) -> logging.Logger:
    logging.basicConfig(level=level.upper(), format="%(message)s")
    return logging.getLogger("sovereign_workbench")


def log_event(logger: logging.Logger, event_name: str, **payload: Any) -> None:
    logger.info(json.dumps({"event": event_name, **redact(payload)}, default=str))
