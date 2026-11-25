from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from app.errors import LLMUnavailableError

logger = logging.getLogger("sirus.llm.ollama")


def _normalise_model_name(name: str) -> str:
    return name[:-7] if name.endswith(":latest") else name


class ModelUnavailableError(LLMUnavailableError):
    """Raised when the configured local Ollama service cannot generate."""

    default_detail = "The local model service is unavailable."


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    async def close(self) -> None:
        """No persistent connection is held; kept for a uniform lifecycle."""
        return None

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=min(self.timeout, 10.0),
                transport=self.transport,
            ) as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError, OSError) as error:
            logger.warning("Ollama model listing failed: %s", error)
            raise ModelUnavailableError("The local model service is not reachable.") from error
        models = data.get("models", []) if isinstance(data, dict) else []
        return [item.get("name", "") for item in models if isinstance(item, dict)]

    async def health(self, required_model: str | None = None) -> Dict[str, Any]:
        try:
            model_names = await self.list_models()
        except ModelUnavailableError as error:
            return {"status": "unhealthy", "detail": error.detail}
        if required_model:
            normalised = {_normalise_model_name(name) for name in model_names}
            if _normalise_model_name(required_model) not in normalised:
                return {
                    "status": "unhealthy",
                    "model": required_model,
                    "detail": f"model '{required_model}' is not installed",
                }
        return {
            "status": "healthy",
            "model": required_model,
            "available_models": model_names,
        }

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = await client.post("/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()

            generated_text = data["response"]
            if not isinstance(generated_text, str):
                raise ValueError("Ollama response was not text")
            return generated_text

        except (httpx.HTTPError, KeyError, TypeError, ValueError, OSError) as error:
            logger.error("Ollama generation failed: %s", error)
            raise ModelUnavailableError() from error
