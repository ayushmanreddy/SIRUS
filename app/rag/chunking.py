from __future__ import annotations

from typing import List


class ChunkMetadata:
    """Metadata preserved for each chunk."""

    def __init__(
        self,
        page_number: int,
        chunk_index: int,
    ) -> None:
        self.page_number = page_number
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, int]:
        return {
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    page_number: int = 0,
) -> List[tuple[str, ChunkMetadata]]:
    """Split text into fixed-size character chunks with overlap.

    Pure function, no I/O. Each chunk preserves its page number and
    sequential chunk_index within that page.

    Args:
        text: The text to chunk.
        chunk_size: Number of characters per chunk.
        chunk_overlap: Number of characters overlap between adjacent chunks.
        page_number: The page number this text came from.

    Returns:
        List of (chunk_text, ChunkMetadata) tuples.
    """
    if not text or not text.strip():
        return []

    chunks: List[tuple[str, ChunkMetadata]] = []
    start = 0
    index = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]
        metadata = ChunkMetadata(page_number=page_number, chunk_index=index)
        chunks.append((chunk_text, metadata))
        index += 1
        start += chunk_size - chunk_overlap

        if end == text_len:
            break

    return chunks