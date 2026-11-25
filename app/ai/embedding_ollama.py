from __future__ import annotations

import logging
import math
from typing import Any, Dict, List

import httpx

from app.ai._embedding import EmbeddingProvider
from app.errors import EmbeddingResponseError, EmbeddingUnavailableError

logger = logging.getLogger("sirus.embedding.ollama")


def _normalise_model_name(name: str) -> str:
    return name[:-7] if name.endswith(":latest") else name


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama-backed embedding provider.

    Calls ``POST {base_url}/api/embed``. The model is configured through
    ``EMBEDDING_MODEL`` (default ``nomic-embed-text``).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str | None = None,
        dimensions: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name or "nomic-embed-text"
        self._dimensions = dimensions
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._transport = transport

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            if self._transport:
                self._client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
            else:
                self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def health(self) -> Dict[str, Any]:
        """Check that Ollama is reachable and the configured model is installed."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [item.get("name", "") for item in models]
            normalised = {_normalise_model_name(name) for name in model_names}
            if _normalise_model_name(self._model_name) not in normalised:
                return {
                    "status": "unhealthy",
                    "model": self._model_name,
                    "detail": f"embedding model '{self._model_name}' is not installed",
                }
            return {
                "status": "healthy",
                "model": self._model_name,
                "available_models": model_names,
            }
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("Ollama embedding health check failed: %s", error)
            return {"status": "unhealthy", "model": self._model_name, "detail": "service not reachable"}

    def _validate_embedding(self, embedding: Any) -> List[float]:
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingResponseError("The embedding service returned an empty vector.")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as error:
            raise EmbeddingResponseError("The embedding service returned invalid numeric values.") from error
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingResponseError("The embedding service returned non-finite values.")
        if self._dimensions is not None and len(vector) != self._dimensions:
            raise EmbeddingResponseError(
                f"Embedding dimension {len(vector)} differs from the configured "
                f"dimension {self._dimensions}. Check EMBEDDING_MODEL and the Qdrant collection."
            )
        return vector

    async def _post_embed(self, inputs: Any) -> List[List[float]]:
        client = await self._get_client()
        payload = {"model": self._model_name, "input": inputs}
        try:
            response = await client.post(f"{self._base_url}/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as error:
            logger.error("Ollama embedding request failed: %s", error)
            raise EmbeddingUnavailableError(
                "The embedding service is unavailable or rejected the request."
            ) from error
        except ValueError as error:
            logger.error("Ollama embedding response was not JSON: %s", error)
            raise EmbeddingResponseError("The embedding service returned invalid JSON.") from error

        embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingResponseError("The embedding service returned no embeddings.")
        return [self._validate_embedding(item) for item in embeddings]

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text string."""
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingResponseError("Cannot embed empty text.")
        embeddings = await self._post_embed(text)
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts in one request."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingResponseError("Cannot embed empty text.")
        embeddings = await self._post_embed(texts)
        if len(embeddings) != len(texts):
            raise EmbeddingResponseError(
                f"Embedding count mismatch: expected {len(texts)}, received {len(embeddings)}."
            )
        dimensions = {len(vector) for vector in embeddings}
        if len(dimensions) != 1:
            raise EmbeddingResponseError("The embedding service returned inconsistent dimensions.")
        return embeddings

    async def get_dimension(self) -> int:
        """Probe the live model for its vector dimension.

        There is intentionally no silent fallback: ingestion must fail loudly
        rather than create a collection with the wrong vector size.
        """
        embedding = await self.embed("dimension-check")
        return len(embedding)

    async def get_model_name(self) -> str:
        return self._model_name
