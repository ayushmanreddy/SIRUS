from __future__ import annotations

import json
import logging
from asyncio import timeout as async_timeout, TimeoutError as AsyncTimeoutError
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from app.config import Settings

from app.ai.model_management import ModelStatus

logger = logging.getLogger("sovereign_workbench.ollama")


class ModelUnavailableError(RuntimeError):
    """Raised when the configured local Ollama service cannot generate."""


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

    def to_metadata(self) -> ModelMetadata:
        from app.ai import ModelMetadata  # lazy import

        return ModelMetadata(
            name=self.name,
            model_type="llm",
            family=self.family,
            context_length=self.context_length,
            requires_gpu=self.size_bytes is not None,
            min_vram_gb=(self.size_bytes / (1024 ** 3) if self.size_bytes else 0.0),
            min_ram_gb=0.0,
        )


class OllamaClient:
    """Local Ollama model client.

    This class provides the inference functionality that was previously
    provided by the OllamaClient implementing ModelProvider. The abstraction
    layer (ModelProvider) is available at app.ai.ModelProvider for future
    implementations.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self._client: httpx.AsyncClient | None = None
        self._current_model: Optional[str] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            )
        return self._client

    async def load(self) -> None:
        """Ollama models are loaded on-demand; no explicit load required."""
        logger.info("Ollama client initialized (models loaded on-demand)")

    async def unload(self) -> None:
        """Unload the current model and release resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        self._current_model = None
        logger.info("Ollama client unloaded")

    async def health(self) -> Dict[str, Any]:
        """Check Ollama service health and available models."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return {
                    "status": "healthy",
                    "service": "ollama",
                    "url": self.base_url,
                    "available_models": [m.get("name", "") for m in models],
                    "model_count": len(models),
                }
            return {"status": "unhealthy", "service": "ollama", "url": self.base_url, "detail": "non-200 response"}
        except (httpx.HTTPError, ConnectionError) as error:
            logger.warning("Ollama health check failed: %s", error)
            return {"status": "unhealthy", "service": "ollama", "url": self.base_url, "detail": str(error)}

    async def generate(self, model: str, prompt: str) -> str:
        """Generate a non-streaming response from the specified model."""
        try:
            async with async_timeout(self.timeout):
                client = await self._get_client()
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                }
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                generated_text = data.get("response", "")
                if not isinstance(generated_text, str):
                    raise ValueError("Ollama response was not text")
                return generated_text
        except (httpx.HTTPError, asyncio.TimeoutError, KeyError, TypeError, ValueError) as error:
            logger.error("Ollama generation failed: %s", error, extra={"model": model})
            raise ModelUnavailableError(
                "Local Ollama service is unavailable or returned an invalid response."
            ) from error

    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        """Stream a response from the specified model."""
        try:
            async with async_timeout(self.timeout):
                client = await self._get_client()
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                }
                async with client.stream("POST", "/api/generate", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                            except json.JSONDecodeError:
                                continue
        except (httpx.HTTPError, AsyncTimeoutError, json.JSONDecodeError) as error:
            logger.error("Ollama streaming failed: %s", error, extra={"model": model})
            raise ModelUnavailableError(
                "Local Ollama service is unavailable during streaming."
            ) from error