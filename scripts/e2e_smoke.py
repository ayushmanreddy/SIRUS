"""Live end-to-end smoke test for a running SIRUS API.

Usage:
    python scripts/e2e_smoke.py --base-url http://localhost:8000

Requires real Qdrant + Ollama + pulled models. Exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    marker = f"SIRUSSMOKE{time.time_ns()}"

    with httpx.Client(base_url=base_url, timeout=args.timeout) as client:
        try:
            live = client.get("/health/live")
            live.raise_for_status()
            print(f"[ok] liveness: {live.json()}")

            ready = client.get("/health/ready")
            print(f"[{'ok' if ready.status_code == 200 else 'fail'}] readiness: {ready.json()}")
            if ready.status_code != 200:
                print("Dependencies are not ready; pull models and start Qdrant/Ollama.")
                return 2
        except httpx.HTTPError as error:
            print(f"[fail] API not reachable: {error}")
            return 2

        text = (
            f"{marker}\n\n"
            f"The SIRUS smoke document code is {marker}. "
            "The designated project lead is Ada Lovelace. "
            "The deployment region is the local workstation."
        )
        ingest = client.post(
            "/documents/ingest",
            files={"file": (f"{marker}.txt", text.encode("utf-8"), "text/plain")},
        )
        if ingest.status_code != 200:
            print(f"[fail] ingest: {ingest.status_code} {ingest.text}")
            return 1
        ingested = ingest.json()
        document_id = ingested["document_id"]
        print(f"[ok] ingested document {document_id} ({ingested['chunk_count']} chunks)")

        query = client.post(
            "/documents/query",
            json={
                "question": "What is the project code and who is the project lead?",
                "top_k": 4,
                "document_id": document_id,
            },
        )
        if query.status_code != 200:
            print(f"[fail] query: {query.status_code} {query.text}")
            return 1
        answer = query.json()
        print(f"[ok] answer: {answer['answer']}")
        print(f"[ok] grounded={answer['grounded']} sources={len(answer['sources'])}")
        if not answer["grounded"] or not answer["sources"]:
            print("[fail] expected a grounded answer with sources")
            return 1
        if not any(source["chunk_id"] for source in answer["sources"]):
            print("[fail] sources are missing chunk IDs")
            return 1
        if marker not in answer["answer"] and "Ada" not in answer["answer"]:
            print("[warn] answer does not contain the expected evidence text")

        unrelated = client.post(
            "/documents/query",
            json={"question": "What is the population of Mars in 3024?", "top_k": 4},
        )
        if unrelated.status_code != 200:
            print(f"[fail] unrelated query: {unrelated.status_code} {unrelated.text}")
            return 1
        unrelated_body = unrelated.json()
        print(f"[ok] unrelated grounded={unrelated_body['grounded']}: {unrelated_body['answer']}")
        if unrelated_body["grounded"]:
            print("[fail] system claimed to answer unrelated question from evidence")
            return 1

    print("[ok] SIRUS live smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
