import asyncio
import json

import httpx

from app.llm_client import OllamaClient


def test_generate_reads_ollama_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body == {"model": "llama3.1:8b", "prompt": "Hello", "stream": False}
        return httpx.Response(200, json={"response": "Local response"})

    client = OllamaClient(
        "http://localhost:11434",
        timeout=10,
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(client.generate("llama3.1:8b", "Hello"))
    assert result == "Local response"