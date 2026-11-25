from __future__ import annotations

import ipaddress
import logging
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("sirus.config")

SUPPORTED_FILE_TYPES = frozenset({"pdf", "txt", "docx"})


class AppProfile(StrEnum):
    DEVELOPMENT = "development"
    DEMO = "demo"
    OFFLINE_DEMO = "offline-demo"


def is_local_or_private_url(value: str) -> bool:
    """Return True only for loopback/private HTTP endpoints used inside deployment."""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {"localhost", "host.docker.internal", "qdrant", "ollama", "llm", "embeddings"}:
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or address.is_private
    except ValueError:
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_profile: AppProfile = AppProfile.DEVELOPMENT
    app_name: str = "SIRUS"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./data/workbench.db"

    # External local services
    llm_service_url: str = Field(default="http://localhost:11434")
    embedding_service_url: str = Field(default="http://localhost:11434")
    vector_store_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = "documents"
    request_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    vector_store_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    generation_timeout_seconds: float = Field(default=120.0, gt=0, le=600)

    # Models
    reasoning_model: str = "llama3.1:8b"
    llm_model: str = "llama3.1:8b"
    coding_model: str = "qwen2.5-coder:7b"
    vision_model: str = "llava:7b"
    embedding_model: str = "nomic-embed-text"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Ingestion / retrieval tuning
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=4000)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    retrieval_top_k: int = Field(default=4, ge=1, le=50)
    retrieval_score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    allowed_file_types: str = "pdf,txt,docx"
    max_upload_mb: int = Field(default=10, gt=0, le=200)
    max_page_count: int = Field(default=100, gt=0)
    max_extracted_chars: int = Field(default=5_000_000, gt=0)
    max_chunks_per_document: int = Field(default=2000, gt=0)

    # OCR fallback for image-only PDF pages
    ocr_enabled: bool = True
    ocr_languages: str = "en"
    ocr_dpi: int = Field(default=200, ge=72, le=600)
    pdf_native_text_min_chars: int = Field(default=40, ge=0)

    # Local model artifact verification (legacy/optional)
    model_artifact_root: str = "./models"
    model_manifest_path: str = "./models/manifest.json"

    @field_validator("llm_service_url", "embedding_service_url", "vector_store_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("qdrant_collection")
    @classmethod
    def validate_collection_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not cleaned.replace("_", "").replace("-", "").isalnum():
            raise ValueError("must contain only letters, digits, '-' and '_'")
        return cleaned

    @field_validator("allowed_file_types")
    @classmethod
    def validate_allowed_file_types(cls, value: str) -> str:
        types = {item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()}
        if not types:
            raise ValueError("at least one file type must be configured")
        unsupported = types - SUPPORTED_FILE_TYPES
        if unsupported:
            raise ValueError(
                f"unsupported configured file types: {', '.join(sorted(unsupported))}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_FILE_TYPES))}"
            )
        return ",".join(sorted(types))

    @model_validator(mode="after")
    def validate_rag_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not self.database_url.startswith("sqlite:///"):
            raise ValueError("SIRUS V1 supports a local SQLite database only")
        return self._validate_offline_endpoints()

    def _validate_offline_endpoints(self) -> "Settings":
        if self.app_profile == AppProfile.OFFLINE_DEMO:
            restricted = {
                "LLM_SERVICE_URL": self.llm_service_url,
                "EMBEDDING_SERVICE_URL": self.embedding_service_url,
                "VECTOR_STORE_URL": self.vector_store_url,
            }
            invalid = [name for name, value in restricted.items() if not is_local_or_private_url(value)]
            if invalid:
                raise ValueError(
                    "offline-demo accepts only loopback/private service URLs: "
                    + ", ".join(invalid)
                    + ". Set APP_PROFILE=development or demo for external endpoints."
                )
        return self

    @property
    def allowed_extensions(self) -> frozenset[str]:
        return frozenset(item.strip() for item in self.allowed_file_types.split(",") if item.strip())

    @property
    def ocr_language_list(self) -> list[str]:
        return [item.strip() for item in self.ocr_languages.split(",") if item.strip()] or ["en"]

    @property
    def answer_model(self) -> str:
        """Model used for grounded RAG answers."""
        return self.llm_model or self.reasoning_model

    @property
    def database_path(self) -> str:
        if not self.database_url.startswith("sqlite:///"):
            raise ValueError("SIRUS V1 supports a local SQLite database only")
        return self.database_url.removeprefix("sqlite:///")
