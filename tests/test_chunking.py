from app.rag.chunking import chunk_text, ChunkMetadata


def test_chunk_text_overlap():
    text = "A" * 200
    chunks, metadatas = [], []
    for chunk, meta in chunk_text(text, chunk_size=50, chunk_overlap=10, page_number=1):
        chunks.append(chunk)
        metadatas.append(meta)
    assert len(chunks) > 1
    # Verify overlap: last chars of one chunk should match first chars of next
    for i in range(len(chunks) - 1):
        assert chunks[i][-10:] == chunks[i + 1][:10], f"Overlap mismatch at chunk {i}"


def test_chunk_text_page_number():
    text = "Page content here."
    chunks, metadatas = [], []
    for chunk, meta in chunk_text(text, chunk_size=50, chunk_overlap=10, page_number=3):
        chunks.append(chunk)
        metadatas.append(meta)
    for meta in metadatas:
        assert meta.page_number == 3


def test_chunk_text_chunk_index():
    text = "A" * 100
    chunks, metadatas = [], []
    for chunk, meta in chunk_text(text, chunk_size=30, chunk_overlap=4, page_number=1):
        chunks.append(chunk)
        metadatas.append(meta)
    for i, meta in enumerate(metadatas):
        assert meta.chunk_index == i


def test_chunk_text_empty_input():
    chunks, metadatas = [], []
    for chunk, meta in chunk_text("", chunk_size=50, chunk_overlap=10, page_number=1):
        chunks.append(chunk)
        metadatas.append(meta)
    assert len(chunks) == 0