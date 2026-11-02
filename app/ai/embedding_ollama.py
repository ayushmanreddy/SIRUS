from __future__ import annotations

import logging
from typing import Any, List

import httpx

from app.ai._embedding import EmbeddingProvider

logger = logging.getLogger("sovereign_workbench.embedding.ollama")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama-backed embedding provider.

    Generates text embeddings by calling the local Ollama embedding API:
    POST {embedding_service_url}/api/embed

    The model name is configured via Settings (default: "nomic-embed-text").
    embed_batch sends multiple texts in one request.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str | None = None,
        dimensions: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name or "nomic-embed-text"
        self._dimensions = dimensions
        self._client: httpx.AsyncClient | None = None
        self._transport = transport

    @property
    def model_name(self) -> str:
        return self._model_name

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            if self._transport:
                self._client = httpx.AsyncClient(timeout=30.0, transport=self._transport)
            else:
                self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def health(self) -> Dict[str, Any]:
        """Check Ollama embedding provider health."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self._base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return {
                    "status": "healthy",
                    "model": self._model_name,
                    "available_models": model_names,
                }
            return {"status": "unhealthy", "detail": f"status {response.status_code}"}
        except (httpx.HTTPError, ConnectionError) as error:
            logger.warning("Ollama embedding health check failed: %s", error)
            return {"status": "unhealthy", "detail": str(error)}

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding for a single text string.

        Calls Ollama /api/embed with payload {"model": ..., "input": text}.
        Reads data["embeddings"][0] from the response.
        """
        client = await self._get_client()
        payload = {"model": self._model_name, "input": text}
        try:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("No embeddings in Ollama response")
            embedding = embeddings[0]
            if self._dimensions is not None and len(embedding) != self._dimensions:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} differs from configured "
                    f"dimensions {self._dimensions}"
                )
            if self._dimensions is not None and len(embedding) > self._dimensions:
                embedding = embedding[: self._dimensions]
            return embedding
        except (httpx.HTTPError, ValueError) as error:
            logger.error("Ollama embedding failed: %s", error)
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text strings.

        Sends all texts in one POST request to /api/embed.
        """
        if not texts:
            return []
        client = await self._get_client()
        payload = {"model": self._model_name, "input": texts}
        try:
            response = await client.post(
                f"{self._base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings")
            if embeddings is None:
                raise ValueError("No embeddings in Ollama response")
            result: List[List[float]] = []
            for emb in embeddings:
                if self._dimensions is not None and len(emb) != self._dimensions:
                    raise ValueError(
                        f"Embedding dimension {len(emb)} differs from configured "
                        f"dimensions {self._dimensions}"
                    )
                if self._dimensions is not None and len(emb) > self._dimensions:
                    emb = emb[: self._dimensions]
                result.append(emb)
            return result
        except (httpx.HTTPError, ValueError) as error:
            logger.error("Ollama embedding batch failed: %s", error)
            raise

    async def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        try:
            emb = await self.embed("dimension-check")
            return len(emb)
        except Exception:
            return self._dimensions or 768

    async def get_model_name(self) -> str:
        """Return the name of the currently configured model."""
        return self._model_name