from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

_DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    collection_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    chunks_json TEXT NOT NULL DEFAULT '[]'
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_sha_model_collection
    ON documents (sha256, embedding_model, collection_name);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at);
"""

_COLLECTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS collection_state (
    collection_name TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    content_type: str
    sha256: str
    size_bytes: int
    page_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int
    collection_name: str
    created_at: str
    chunks: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class CollectionState:
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    created_at: str


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    try:
        chunks = json.loads(row["chunks_json"] or "[]")
    except json.JSONDecodeError:
        chunks = []
    return DocumentRecord(
        document_id=row["document_id"],
        filename=row["filename"],
        content_type=row["content_type"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        embedding_model=row["embedding_model"],
        embedding_dimension=row["embedding_dimension"],
        collection_name=row["collection_name"],
        created_at=row["created_at"],
        chunks=chunks,
    )


class DocumentRegistry:
    """SQLite-backed metadata registry for indexed documents.

    Qdrant stores vectors and chunk payloads; this registry stores document-level
    identity, duplicate-detection hashes, and collection compatibility metadata.
    Keeping it separate makes listing documents and idempotent ingestion cheap.
    """

    def __init__(self, database_url: str) -> None:
        if not database_url.startswith("sqlite:///"):
            raise ValueError("SIRUS V1 supports a local SQLite database only")
        self.path = Path(database_url.removeprefix("sqlite:///"))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(_DOCUMENTS_SCHEMA)
            connection.executescript(_COLLECTION_SCHEMA)

    def find_duplicate(
        self,
        sha256: str,
        embedding_model: str,
        collection_name: str,
    ) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT * FROM documents
                   WHERE sha256 = ? AND embedding_model = ? AND collection_name = ?""",
                (sha256, embedding_model, collection_name),
            ).fetchone()
        return _row_to_record(row) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def add(self, record: DocumentRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO documents (
                    document_id, filename, content_type, sha256, size_bytes,
                    page_count, chunk_count, embedding_model, embedding_dimension,
                    collection_name, created_at, chunks_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.document_id,
                    record.filename,
                    record.content_type,
                    record.sha256,
                    record.size_bytes,
                    record.page_count,
                    record.chunk_count,
                    record.embedding_model,
                    record.embedding_dimension,
                    record.collection_name,
                    record.created_at,
                    json.dumps(record.chunks, sort_keys=True),
                ),
            )

    def list_documents(self, limit: int = 50, offset: int = 0) -> tuple[list[DocumentRecord], int]:
        with self._connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_record(row) for row in rows], int(total)

    def count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def get_collection_state(self, collection_name: str) -> CollectionState | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM collection_state WHERE collection_name = ?",
                (collection_name,),
            ).fetchone()
        if not row:
            return None
        return CollectionState(
            collection_name=row["collection_name"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            created_at=row["created_at"],
        )

    def set_collection_state(
        self,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO collection_state (
                    collection_name, embedding_model, embedding_dimension, created_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(collection_name) DO UPDATE SET
                    embedding_model = excluded.embedding_model,
                    embedding_dimension = excluded.embedding_dimension""",
                (
                    collection_name,
                    embedding_model,
                    embedding_dimension,
                    datetime.now(UTC).isoformat(),
                ),
            )

