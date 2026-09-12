"""Local embedding infrastructure abstraction.

Defines the interface for generating text embeddings using local models.
Follows the same pattern as ModelProvider for consistency and extensibility.

Embedding models convert text into vector representations suitable for
similarity search in the vector store. All processing stays on-premise;
no external embedding APIs are used.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sovereign_workbench.embedding")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding model providers."""

    @abstractmethod
    async def health(self) -> Dict[str, Any]:
        """Check embedding provider health status."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings."""
        ...

    @abstractmethod
    async def get_dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    async def get_model_name(self) -> str:
        """Return the name of the currently configured model."""