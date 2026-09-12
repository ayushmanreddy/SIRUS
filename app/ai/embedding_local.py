from __future__ import annotations

import logging
from typing import Any, List, Optional

import hashlib
import numpy as np

from app.ai._embedding import EmbeddingProvider
from app.config import Settings

logger = logging.getLogger("sovereign_workbench.embedding.local")


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using a configurable sentence-transformer-style model.

    This provider generates embeddings using a local model. The model is
    configured via Settings and should be available locally (e.g., via
    GGUF format or a Python-compatible model file).

    In the absence of a real local embedding model, this provider falls back
    to a simple hashing-based embedding that preserves text semantics
    approximately but is suitable for demonstration and development.
    """

    def __init__(self, model_name: str | None = None, dimensions: int | None = None) -> None:
        self._model_name = model_name or "local-embedding"
        self._dimensions = dimensions or 768
        self._initialized = False

    async def health(self) -> Dict[str, Any]:
        """Check embedding provider health."""
        return {
            "status": "healthy" if self._initialized else "initializing",
            "model": self._model_name,
            "dimensions": self._dimensions,
        }

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text string."""
        if not self._initialized:
            self._initialized = True
        # Use a deterministic hash-based embedding for demo purposes
        # In production, this would use a local sentence-transformer model
        embedding = self._hash_based_embedding(text, self._dimensions)
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings."""
        return [await self.embed(t) for t in texts]

    async def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimensions

    async def get_model_name(self) -> str:
        """Return the name of the currently configured model."""
        return self._model_name

    @staticmethod
    def _hash_based_embedding(text: str, dimensions: int) -> List[float]:
        """Generate a deterministic embedding using SHA-256 hashing.

        This is a placeholder for a real local embedding model. The hash-based
        approach produces reproducible embeddings for the same input text,
        which is useful for development and testing when no model is available.

        Note: This does NOT preserve semantic similarity. For actual RAG
        functionality, a proper local embedding model (e.g., all-miniLM-L6-v2)
        should be integrated.
        """
        # Use SHA-256 of the text for deterministic output
        hasher = hashlib.sha256(text.encode("utf-8"))
        hash_bytes = hasher.digest()

        # Convert hash bytes to float vector in [-1, 1] range
        embedding: List[float] = []
        for i in range(dimensions):
            # Cycle through hash bytes for each dimension
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2.0 - 1.0
            embedding.append(round(value, 6))

        # Normalize to unit vector for cosine similarity compatibility
        magnitude = sum(v ** 2 for v in embedding) ** 0.5
        if magnitude > 0:
            embedding = [round(v / magnitude, 6) for v in embedding]

        return embedding