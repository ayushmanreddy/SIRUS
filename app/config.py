from __future__ import annotations

import ipaddress
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    if host in {"localhost", "host.docker.internal", "qdrant", "llm", "embeddings"}:
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or address.is_private
    except ValueError:
        return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_profile: AppProfile = AppProfile.DEVELOPMENT
    app_name: str = "Sovereign AI Workbench"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./data/workbench.db"
    llm_service_url: str = "http://localhost:11434"
    embedding_service_url: str = "http://localhost:11434"
    vector_store_url: str = "http://localhost:6333"

    reasoning_model: str = "llama3.1:8b"
    coding_model: str = "qwen2.5-coder:7b"
    vision_model: str = "llava:7b"

    model_artifact_root: str = "./models"
    model_manifest_path: str = "./models/manifest.json"
    request_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    generation_timeout_seconds: float = Field(default=120.0, gt=0, le=600)

    @field_validator("llm_service_url", "embedding_service_url", "vector_store_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def block_external_ai_endpoints_offline(self) -> "Settings":
        if self.app_profile == AppProfile.OFFLINE_DEMO:
            restricted = {
                "LLM_SERVICE_URL": self.llm_service_url,
                "EMBEDDING_SERVICE_URL": self.embedding_service_url,
                "VECTOR_STORE_URL": self.vector_store_url,
            }
            invalid = [name for name, value in restricted.items() if not is_local_or_private_url(value)]
            if invalid:
                raise ValueError("offline-demo accepts only loopback/private service URLs: " + ", ".join(invalid))
        return self
    