from __future__ import annotations

import io

import fitz
import pytest

import app.ocr as ocr_module
from app.config import Settings
from app.documents.extraction import extract_document, sanitize_filename
from app.errors import (
    DocumentTooLargeToIndexError,
    InvalidDocumentError,
    NoExtractableTextError,
    OCRUnavailableError,
    UnsupportedFileTypeError,
)
from app.ocr import OCRExtractionError


def _settings(**overrides) -> Settings:
    values = {
        "database_url": "sqlite:///./unused.db",
        "ocr_enabled": False,
        "pdf_native_text_min_chars": 10,
        "max_page_count": 5,
        "max_upload_mb": 2,
        "max_extracted_chars": 1_000_000,
        "allowed_file_types": "pdf,txt,docx",
        "chunk_size": 200,
        "chunk_overlap": 20,
    }
    values.update(overrides)
    return Settings(**values)


def _pdf_with_text(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=16)
    data = document.write()
    document.close()
    return data


def _blank_pdf() -> bytes:
    document = fitz.open()
    document.new_page()
    data = document.write()
    document.close()
    return data


def test_extract_plain_text_file():
    data = "SIRUS knowledge base\n\nSecond paragraph.".encode("utf-8")
    result = extract_document("notes.txt", "text/plain", data, _settings())
    assert result.filename == "notes.txt"
    assert result.file_type == "txt"
    assert result.page_count == 1
    assert "SIRUS knowledge base" in result.text
    assert len(result.sha256) == 64


def test_extract_text_file_with_cp1252_fallback():
    data = "Caf\xe9 report".encode("cp1252")
    result = extract_document("report.txt", "text/plain", data, _settings())
    assert "Caf" in result.text


def test_extract_pdf_uses_native_text():
    result = extract_document(
        "report.pdf", "application/pdf", _pdf_with_text("SIRUS native PDF text"), _settings()
    )
    assert "SIRUS native PDF text" in result.text
    assert result.pages[0].method == "native"


def test_extract_docx_paragraphs_and_tables():
    from docx import Document

    document = Document()
    document.add_paragraph("Policy paragraph")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Owner"
    table.rows[0].cells[1].text = "Security"
    buffer = io.BytesIO()
    document.save(buffer)

    result = extract_document("policy.docx", "application/vnd.openxmlformats", buffer.getvalue(), _settings())
    assert "Policy paragraph" in result.text
    assert "Owner | Security" in result.text
    assert result.page_count == 1


def test_ocr_fallback_for_blank_pdf(monkeypatch):
    class FakeReader:
        def readtext(self, image, detail=0):
            return ["Scanned page text"]

    monkeypatch.setattr(ocr_module, "_get_reader", lambda languages=None: FakeReader())
    settings = _settings(ocr_enabled=True)
    result = extract_document("scan.pdf", "application/pdf", _blank_pdf(), settings)
    assert "Scanned page text" in result.text
    assert result.pages[0].method == "ocr"


def test_ocr_unavailable_is_reported(monkeypatch):
    def raise_ocr(languages=None):
        raise OCRExtractionError("no OCR")

    monkeypatch.setattr(ocr_module, "_get_reader", raise_ocr)
    settings = _settings(ocr_enabled=True)
    with pytest.raises(OCRUnavailableError):
        extract_document("scan.pdf", "application/pdf", _blank_pdf(), settings)


def test_image_only_pdf_without_ocr_has_no_text():
    with pytest.raises(NoExtractableTextError):
        extract_document("scan.pdf", "application/pdf", _blank_pdf(), _settings(ocr_enabled=False))


def test_empty_file_is_rejected():
    with pytest.raises(InvalidDocumentError):
        extract_document("empty.txt", "text/plain", b"", _settings())


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        extract_document("malware.exe", "application/octet-stream", b"binary", _settings())


def test_corrupted_pdf_is_rejected():
    with pytest.raises(InvalidDocumentError):
        extract_document("broken.pdf", "application/pdf", b"this is not a pdf", _settings())


def test_pdf_page_limit_is_enforced():
    document = fitz.open()
    for _ in range(3):
        document.new_page()
    data = document.write()
    document.close()
    with pytest.raises(DocumentTooLargeToIndexError):
        extract_document("large.pdf", "application/pdf", data, _settings(max_page_count=2))


def test_extracted_text_limit_is_enforced():
    data = b"x" * 5000
    with pytest.raises(DocumentTooLargeToIndexError):
        extract_document("big.txt", "text/plain", data, _settings(max_extracted_chars=1000))


def test_filename_traversal_is_reduced_to_basename():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename(r"C:\Users\secret\report.pdf") == "report.pdf"
    assert sanitize_filename(None) == "document"
