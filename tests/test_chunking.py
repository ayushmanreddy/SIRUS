from __future__ import annotations

import pytest

from app.rag.chunking import chunk_text, normalize_text


def test_chunk_text_overlap():
    text = "A" * 200
    chunks = [chunk for chunk, _ in chunk_text(text, chunk_size=50, chunk_overlap=10, page_number=1)]
    assert len(chunks) > 1
    for index in range(len(chunks) - 1):
        assert chunks[index][-10:] == chunks[index + 1][:10], f"Overlap mismatch at chunk {index}"


def test_chunk_text_page_and_index_metadata():
    text = "Page content here. " * 20
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10, page_number=3)
    assert chunks
    for index, (_, metadata) in enumerate(chunks):
        assert metadata.page_number == 3
        assert metadata.chunk_index == index
        assert metadata.char_start is not None
        assert metadata.char_end is not None


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("", chunk_size=50, chunk_overlap=10, page_number=1) == []
    assert chunk_text("   \n\n  ", chunk_size=50, chunk_overlap=10, page_number=1) == []


def test_chunk_text_prefers_natural_boundaries():
    text = "First sentence ends here. Second sentence continues. Third sentence finishes the paragraph."
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=5, page_number=1)
    assert len(chunks) > 1
    assert chunks[0][0].rstrip().endswith(".")


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0, chunk_overlap=0)


def test_chunk_text_handles_large_text():
    text = "word " * 20000
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=100, page_number=1)
    assert len(chunks) > 100
    assert all(len(chunk) <= 800 for chunk, _ in chunks)


def test_normalize_text_preserves_paragraphs_and_removes_control_chars():
    raw = "Hello\r\n\r\n\r\nworld\x00!\x0c Next\tline   "
    normalized = normalize_text(raw)
    assert "\r" not in normalized
    assert "\x00" not in normalized
    assert "Hello\n\nworld" in normalized
