from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import logging

logger = logging.getLogger("sovereign_workbench.model_registry")


@dataclass
class ModelStatus:
    """Current status of a model."""
    model_name: str
    is_loaded: bool = False
    is_healthy: bool = False
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    requests_served: int = 0


@dataclass
class ModelRegistry:
    """Registry managing available models and the active model."""

    _providers: Dict[str, Any] = field(default_factory=dict)
    _active_model: Optional[str] = None
    _model_status: Dict[str, ModelStatus] = field(default_factory=dict)

    def register(self, name: str, provider: Any) -> None:
        """Register a model provider."""
        self._providers[name] = provider
        self._model_status[name] = ModelStatus(model_name=name)

    def unregister(self, name: str) -> None:
        """Unregister a model provider."""
        self._providers.pop(name, None)
        self._model_status.pop(name, None)
        if self._active_model == name:
            self._active_model = None

    def set_active(self, name: str) -> None:
        """Set the active model by name."""
        if name not in self._providers:
            raise ValueError(f"Model '{name}' is not registered")
        self._active_model = name
        self._model_status[name].is_loaded = True

    def get_active(self) -> Optional[str]:
        """Get the name of the active model."""
        return self._active_model

    def get_status(self, name: Optional[str] = None) -> ModelStatus:
        """Get the status of a model (or the active model if name not specified)."""
        target = name or self._active_model
        if target is None or target not in self._model_status:
            return ModelStatus(model_name="none")
        return self._model_status[target]

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._providers.keys())

    def get_metadata(self, name: str) -> Optional[object]:
        """Get metadata for a specific model."""
        from app.ai import ModelMetadata  # lazy import to avoid circular dependency
        if name not in self._model_status:
            return None
        return ModelMetadata(
            name=name,
            model_type="llm",
            family=None,
            context_length=None,
            requires_gpu=False,
            min_vram_gb=0.0,
            min_ram_gb=0.0,
        )