from __future__ import annotations

import ipaddress
import logging
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger("sovereign_workbench.config")


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


def check_service_not_external(value: str, setting_name: str) -> str:
    """Validate that a service URL is local/private; raise ValueError if not."""
    if not is_local_or_private_url(value):
        raise ValueError(
            f"{setting_name} must be a loopback or private network URL "
            f"when APP_PROFILE=offline-demo. Got: {value}"
        )
    return value.rstrip("/")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_profile: AppProfile = AppProfile.DEVELOPMENT
    app_name: str = "Sovereign AI Workbench"
    app_version: str = "0.1.0"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./data/workbench.db"
    llm_service_url: str = Field(default="http://localhost:11434", validator=False)
    embedding_service_url: str = Field(default="http://localhost:11434", validator=False)
    vector_store_url: str = Field(default="http://localhost:6333", validator=False)

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
                raise ValueError(
                    "offline-demo accepts only loopback/private service URLs: "
                    + ", ".join(invalid)
                    + ". Set APP_PROFILE=development or demo for external endpoints."
                )
        return self
    