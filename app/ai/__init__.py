from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from app.ai.model_management import ModelRegistry, ModelStatus
from app.ai._embedding import EmbeddingProvider
from app.ai._vector_store import VectorStoreInterface, VectorStoreMetadata, VectorStoreSearchResult
from app.ai.ollama_client import OllamaClient, ModelUnavailableError

logger = logging.getLogger("sovereign_workbench.ai")


class ModelProvider(ABC):
    """Abstract base class for AI model providers."""

    @abstractmethod
    async def load(self) -> None:
        """Load the model into memory/initialize the provider."""

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model from memory/release resources."""

    @abstractmethod
    async def health(self) -> Dict[str, Any]:
        """Check model provider health status."""

    @abstractmethod
    async def generate(self, model: str, prompt: str) -> str:
        """Generate a response from the model."""

    @abstractmethod
    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        """Stream a response from the model."""


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