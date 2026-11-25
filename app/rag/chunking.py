from __future__ import annotations

import re
from typing import Dict, List

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_text(text: str) -> str:
    """Normalize extraction output without destroying paragraph structure.

    - Unifies line endings and removes form-feed/control characters.
    - Collapses repeated horizontal whitespace and blank-line runs.
    - Keeps paragraph breaks so the chunker can prefer semantic boundaries.
    """
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


class ChunkMetadata:
    """Metadata preserved for each chunk."""

    def __init__(
        self,
        page_number: int,
        chunk_index: int,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> None:
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.char_start = char_start
        self.char_end = char_end

    def to_dict(self) -> Dict[str, int | None]:
        return {
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def _find_boundary(text: str, start: int, target_end: int) -> int:
    """Choose a natural boundary near the target end of a chunk.

    Paragraph boundaries are preferred, then sentence endings, then word
    boundaries. This keeps retrieved passages readable without a heavyweight
    recursive splitter. If nothing is found the hard character boundary wins.
    """
    if target_end >= len(text):
        return len(text)

    search_from = start + max(1, int((target_end - start) * 0.6))
    window = text[search_from:target_end]

    for marker in ("\n\n", "\n"):
        position = window.rfind(marker)
        if position != -1:
            return search_from + position + len(marker)

    for position in range(len(window) - 1, -1, -1):
        if window[position] in ".!?":
            return search_from + position + 1

    for position in range(len(window) - 1, -1, -1):
        if window[position].isspace():
            return search_from + position

    return target_end


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    page_number: int = 0,
) -> List[tuple[str, ChunkMetadata]]:
    """Split text into overlapping, boundary-aware chunks.

    Chunk size and overlap are measured in characters because OCR output is
    tokenized differently across local models. Character sizing is deterministic
    across embedding models and avoids a tokenizer dependency.

    Trade-offs:
    - Natural boundaries make each passage more readable.
    - A fixed character budget keeps chunks within embedding context limits.
    - Overlap preserves context that straddles a boundary at the cost of
      duplicate text (and slightly more embedding work).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: List[tuple[str, ChunkMetadata]] = []
    start = 0
    index = 0
    text_len = len(normalized)
    min_tail = min(50, max(1, chunk_size // 4))

    while start < text_len:
        target_end = min(start + chunk_size, text_len)
        end = _find_boundary(normalized, start, target_end)
        if end <= start:
            end = target_end
        chunk = normalized[start:end]
        chunks.append(
            (
                chunk,
                ChunkMetadata(
                    page_number=page_number,
                    chunk_index=index,
                    char_start=start,
                    char_end=end,
                ),
            )
        )
        index += 1

        if end >= text_len:
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    # Avoid a tiny trailing fragment that carries little retrievable context.
    if len(chunks) > 1:
        last_text, _ = chunks[-1]
        if len(last_text.strip()) < min_tail:
            previous_text, previous_meta = chunks[-2]
            tail_text, _ = chunks[-1]
            chunks[-2] = (
                previous_text + tail_text,
                ChunkMetadata(
                    page_number=previous_meta.page_number,
                    chunk_index=previous_meta.chunk_index,
                    char_start=previous_meta.char_start,
                    char_end=previous_meta.char_end,
                ),
            )
            chunks.pop()

    return chunks
