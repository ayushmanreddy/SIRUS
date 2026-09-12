from __future__ import annotations
from typing import Any, Dict, Optional


class ModelMetadata:
    """Metadata describing a model available for use."""

    def __init__(
        self,
        name: str,
        model_type: str,
        family: str | None = None,
        context_length: int | None = None,
        requires_gpu: bool = False,
        min_vram_gb: float = 0.0,
        min_ram_gb: float = 0.0,
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.family = family
        self.context_length = context_length
        self.requires_gpu = requires_gpu
        self.min_vram_gb = min_vram_gb
        self.min_ram_gb = min_ram_gb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "family": self.family,
            "context_length": self.context_length,
            "requires_gpu": self.requires_gpu,
            "min_vram_gb": self.min_vram_gb,
            "min_ram_gb": self.min_ram_gb,
        }