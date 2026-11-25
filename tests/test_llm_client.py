from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.llm_client import ModelUnavailableError, OllamaClient


def test_generate_reads_ollama_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body == {"model": "llama3.1:8b", "prompt": "Hello", "stream": False}
        return httpx.Response(200, json={"response": "Local response"})

    client = OllamaClient(
        "http://localhost:11434", timeout=10, transport=httpx.MockTransport(handler)
    )
    assert asyncio.run(client.generate("llama3.1:8b", "Hello")) == "Local response"


def test_generate_sends_system_prompt_and_temperature():
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"response": "Grounded answer [1]."})

    client = OllamaClient(
        "http://localhost:11434", timeout=10, transport=httpx.MockTransport(handler)
    )
    import asyncio

    answer = asyncio.run(
        client.generate("llama3.1:8b", "Question", system="Use evidence.", temperature=0.1)
    )
    assert answer == "Grounded answer [1]."
    assert captured["system"] == "Use evidence."
    assert captured["options"] == {"temperature": 0.1}


def test_generate_wraps_upstream_failure():
    def blocked(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = OllamaClient("http://localhost:11434", timeout=1, transport=httpx.MockTransport(blocked))
    import asyncio

    with pytest.raises(ModelUnavailableError):
        asyncio.run(client.generate("llama3.1:8b", "Hello"))


@pytest.mark.asyncio
async def test_health_checks_required_model():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})

    client = OllamaClient("http://localhost:11434", timeout=1, transport=httpx.MockTransport(handler))
    assert (await client.health("llama3.1:8b"))["status"] == "healthy"
    missing = await client.health("does-not-exist")
    assert missing["status"] == "unhealthy"
    assert "not installed" in missing["detail"]
