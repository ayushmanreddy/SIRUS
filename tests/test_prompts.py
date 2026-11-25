from __future__ import annotations

from app.rag.prompts import (
    NO_EVIDENCE_ANSWER,
    build_rag_prompt,
    is_insufficient_answer,
)
from app.rag.retrieval import RetrievedChunk


def _chunk(citation_id: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        citation_id=citation_id,
        document_id="doc-1",
        document_name="manual.pdf",
        chunk_id="chunk-1",
        page_number=2,
        chunk_index=0,
        content="Paris is the capital of France.",
        score=0.9,
    )


def test_prompt_contains_grounding_rules_and_evidence():
    system, user = build_rag_prompt("What is the capital of France?", [_chunk()])
    assert "Answer ONLY from the EVIDENCE" in system
    assert "Never invent" in system
    assert NO_EVIDENCE_ANSWER in system
    assert "[1] source: manual.pdf (page 2), chunk: chunk-1" in user
    assert "Paris is the capital of France." in user
    assert "QUESTION:" in user


def test_prompt_handles_no_evidence():
    _, user = build_rag_prompt("Question", [])
    assert "(no evidence retrieved)" in user


def test_insufficient_answer_detection():
    assert is_insufficient_answer("")
    assert is_insufficient_answer(NO_EVIDENCE_ANSWER)
    assert is_insufficient_answer("Not found in the document.")
    assert not is_insufficient_answer("Paris is the capital [1].")
