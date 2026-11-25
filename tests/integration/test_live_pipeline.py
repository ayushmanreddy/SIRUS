"""Live integration test for the real Ollama + Qdrant pipeline.

Run only when the stack is running:

    $env:SIRUS_INTEGRATION=1
    $env:SIRUS_BASE_URL="http://localhost:8000"
    python -m pytest tests/integration -v

This test is intentionally skipped in normal unit-test runs because mocked
tests cannot prove that embeddings, Qdrant, and the LLM work together.
"""
from __future__ import annotations

import os
import time

import httpx
import pytest

pytestmark = pytest.mark.integration


def _enabled() -> bool:
    return os.getenv("SIRUS_INTEGRATION", "").strip() in {"1", "true", "yes"}


@pytest.mark.skipif(not _enabled(), reason="set SIRUS_INTEGRATION=1 to run live integration tests")
def test_live_ingest_query_and_grounding():
    base_url = os.getenv("SIRUS_BASE_URL", "http://localhost:8000").rstrip("/")
    marker = f"SIRUSLIVE{time.time_ns()}"

    with httpx.Client(base_url=base_url, timeout=180.0) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200, ready.text

        document = (
            f"{marker}\n\n"
            "The SIRUS integration document states that the project code name is "
            f"{marker}. The designated project lead is Ada Lovelace. "
            "The deployment region is the local test workstation."
        )
        ingest = client.post(
            "/documents/ingest",
            files={"file": (f"{marker}.txt", document.encode("utf-8"), "text/plain")},
        )
        assert ingest.status_code == 200, ingest.text
        ingested = ingest.json()
        assert ingested["chunk_count"] >= 1
        document_id = ingested["document_id"]

        query = client.post(
            "/documents/query",
            json={
                "question": "What is the project code name and who is the project lead?",
                "top_k": 4,
                "document_id": document_id,
            },
        )
        assert query.status_code == 200, query.text
        answer = query.json()
        assert answer["grounded"] is True, answer
        assert answer["sources"], answer
        assert marker in answer["answer"] or "Ada" in answer["answer"], answer
        assert {source["document_id"] for source in answer["sources"]} == {document_id}
        assert all(source["chunk_id"] for source in answer["sources"])

        unrelated = client.post(
            "/documents/query",
            json={"question": "What is the population of Mars in 3024?", "top_k": 4},
        )
        assert unrelated.status_code == 200, unrelated.text
        unrelated_body = unrelated.json()
        assert unrelated_body["grounded"] is False, unrelated_body
        assert "could not find enough information" in unrelated_body["answer"].lower()
