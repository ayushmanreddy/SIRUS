"""Validates locally transferred model artifacts; no downloads are performed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from app.config import Settings

class ModelArtifact(BaseModel):
    identifier: str = Field(min_length=1)
    role: str = Field(pattern="^(llm|embedding)$")
    file: str = Field(min_length=1)
    sha256: str = Field(pattern="^[a-fA-F0-9]{64}$")
    runtime: str = Field(min_length=1)
    format: str = Field(min_length=1)
    min_system_ram_gb: int = Field(ge=1)
    min_vram_gb: int = Field(ge=0)
    context_length: int | None = Field(default=None, ge=1)

class ModelManifest(BaseModel):
    version: int = Field(ge=1)
    artifacts: list[ModelArtifact] = Field(min_length=1)

class ArtifactVerification(BaseModel):
    status: str
    detail: str
    models: list[dict[str, str]] = Field(default_factory=list)

def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def verify_model_artifacts(settings: Settings) -> ArtifactVerification:
    manifest_path, root = Path(settings.model_manifest_path), Path(settings.model_artifact_root).resolve()
    if not manifest_path.is_file(): return ArtifactVerification(status="unavailable", detail="local model manifest is missing")
    try: manifest = ModelManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError): return ArtifactVerification(status="unavailable", detail="model manifest is invalid")
    models: list[dict[str, str]] = []
    for artifact in manifest.artifacts:
        candidate = (root / artifact.file).resolve()
        if root not in candidate.parents: return ArtifactVerification(status="unavailable", detail="model manifest contains an unsafe file reference")
        if not candidate.is_file(): return ArtifactVerification(status="unavailable", detail=f"required {artifact.role} artifact is missing")
        if checksum(candidate).lower() != artifact.sha256.lower(): return ArtifactVerification(status="unavailable", detail=f"required {artifact.role} artifact checksum does not match")
        models.append({"identifier": artifact.identifier, "role": artifact.role, "runtime": artifact.runtime, "format": artifact.format})
    return ArtifactVerification(status="ready", detail="all approved local artifacts verified", models=models)
