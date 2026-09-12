from __future__ import annotations

import asyncio
import httpx


class ModelUnavailableError(RuntimeError):
    """Raised when the configured local Ollama service cannot generate."""


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

    async def generate(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

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

        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise ModelUnavailableError(
                "Local Ollama service is unavailable or returned an invalid response."
            ) from error