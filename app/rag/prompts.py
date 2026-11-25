from __future__ import annotations

from typing import Protocol, Sequence

NO_EVIDENCE_ANSWER = (
    "I could not find enough information in the indexed documents to answer that question."
)

SYSTEM_PROMPT = f"""You are SIRUS, a private retrieval-augmented knowledge assistant.

Rules:
1. Answer ONLY from the EVIDENCE supplied by the user. The evidence was retrieved
   from the user's own documents.
2. Never invent document contents, names, numbers, or citations.
3. If the evidence is insufficient, reply exactly with:
   "{NO_EVIDENCE_ANSWER}"
4. Do not use general world knowledge to fill gaps. If you add interpretation,
   label it clearly as inference and explain what evidence supports it.
5. Cite evidence items inline using their bracketed numbers, for example [1].
6. Answer concisely. Do not repeat the evidence verbatim unless it is necessary.
"""


class EvidenceChunk(Protocol):
    citation_id: int
    document_name: str
    page_number: int | None
    chunk_id: str
    content: str


def build_rag_prompt(question: str, evidence: Sequence[EvidenceChunk]) -> tuple[str, str]:
    """Build the system and user prompts for a grounded answer.

    Prompt construction lives outside the API routes so it can be unit tested
    and changed without touching HTTP handling.
    """
    blocks: list[str] = []
    for item in evidence:
        page = f"page {item.page_number}" if item.page_number is not None else "page unknown"
        blocks.append(
            f"[{item.citation_id}] source: {item.document_name} ({page}), "
            f"chunk: {item.chunk_id}\n{item.content}"
        )

    evidence_text = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    user_prompt = (
        f"EVIDENCE:\n{evidence_text}\n\n"
        f"QUESTION:\n{question.strip()}\n\n"
        "Answer using only the evidence above and include inline citations like [1]."
    )
    return SYSTEM_PROMPT, user_prompt


def is_insufficient_answer(answer: str) -> bool:
    """Detect the model's explicit no-evidence response."""
    lowered = " ".join(answer.lower().split())
    markers = (
        "could not find enough information",
        "cannot find enough information",
        "insufficient evidence",
        "not found in the document",
        "not enough information in the indexed documents",
    )
    return not lowered or any(marker in lowered for marker in markers)
