from __future__ import annotations
from typing import Any, Dict, Optional


class OllamaModelInfo:
    """Information about a model available from Ollama."""

    def __init__(
        self,
        name: str,
        size: str | None = None,
        modified_at: str | None = None,
        digest: str | None = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.size = size
        self.modified_at = modified_at
        self.digest = digest
        self.details = details or {}

    @property
    def context_length(self) -> int | None:
        return self.details.get("context_length") if self.details else None

    @property
    def family(self) -> str | None:
        return self.details.get("family") if self.details else None

    @property
    def size_bytes(self) -> Optional[int]:
        if self.size:
            try:
                return int(self.size)
            except ValueError:
                return None
        return None

    def to_metadata(self) -> "ModelMetadata":
        from .ModelMetadata import ModelMetadata  # local import to avoid circular

        return ModelMetadata(
            name=self.name,
            model_type="llm",
            family=self.family,
            context_length=self.context_length,
            requires_gpu=self.size_bytes is not None,
            min_vram_gb=(self.size_bytes / (1024 ** 3) if self.size_bytes else 0.0),
            min_ram_gb=0.0,
        )