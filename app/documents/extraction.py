from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.errors import (
    DocumentTooLargeToIndexError,
    InvalidDocumentError,
    NoExtractableTextError,
    OCRUnavailableError,
    UnsupportedFileTypeError,
)
from app.ocr import (
    InvalidPDFError,
    OCRExtractionError,
    PDFPageLimitError,
    extract_pdf_pages,
)

logger = logging.getLogger("sirus.documents.extraction")

_FILENAME_CLEAN_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    method: str


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str
    content_type: str
    sha256: str
    size_bytes: int
    pages: list[ExtractedPage] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


def sanitize_filename(filename: str | None, fallback: str = "document") -> str:
    """Return a display-safe basename.

    Uploaded filenames are never used as filesystem paths, but they are stored
    in metadata and returned in citations, so traversal components and control
    characters are removed. Only the final path component is kept.
    """
    if not filename:
        return fallback
    cleaned = _FILENAME_CLEAN_RE.sub("", filename.replace("\\", "/"))
    basename = Path(cleaned).name.strip()
    if basename in {"", ".", ".."}:
        return fallback
    return basename[:255]


def _file_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix


def _validate_magic(file_type: str, data: bytes) -> None:
    if file_type == "pdf" and b"%PDF-" not in data[:2048]:
        raise InvalidDocumentError("The uploaded file is not a valid PDF.")
    if file_type == "docx" and not data.startswith(b"PK\x03\x04"):
        raise InvalidDocumentError("The uploaded file is not a valid DOCX file.")
    if file_type == "txt" and b"\x00" in data[:4096]:
        raise InvalidDocumentError("The uploaded text file appears to be binary.")


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InvalidDocumentError("The text file could not be decoded.")


def _extract_pdf(data: bytes, settings: Settings) -> list[ExtractedPage]:
    try:
        pages = extract_pdf_pages(
            data,
            ocr_enabled=settings.ocr_enabled,
            languages=settings.ocr_language_list,
            dpi=settings.ocr_dpi,
            min_native_chars=settings.pdf_native_text_min_chars,
            max_pages=settings.max_page_count,
        )
    except PDFPageLimitError as error:
        raise DocumentTooLargeToIndexError(
            f"The PDF exceeds the configured {settings.max_page_count}-page limit."
        ) from error
    except OCRExtractionError as error:
        raise OCRUnavailableError(
            "This PDF needs OCR, but OCR is unavailable. Enable OCR or upload a text-based PDF."
        ) from error
    except InvalidPDFError as error:
        raise InvalidDocumentError("The uploaded PDF could not be read.") from error

    return [ExtractedPage(page.page_number, page.text, page.method) for page in pages]


def _extract_docx(data: bytes) -> list[ExtractedPage]:
    try:
        from docx import Document
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise UnsupportedFileTypeError("DOCX support is not installed on this server.") from error

    try:
        document = Document(io.BytesIO(data))
    except Exception as error:
        raise InvalidDocumentError("The uploaded DOCX file could not be read.") from error

    parts: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text.strip())
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n\n".join(parts)
    # python-docx does not expose physical page boundaries.
    return [ExtractedPage(page_number=1, text=text, method="docx")]


def extract_document(filename: str | None, content_type: str | None, data: bytes, settings: Settings) -> ExtractedDocument:
    """Validate, classify, and extract text from an uploaded document.

    This function is synchronous and CPU/IO heavy (PDF parsing, optional OCR),
    so callers should run it in a worker thread.
    """
    safe_name = sanitize_filename(filename)
    file_type = _file_extension(safe_name)
    allowed = settings.allowed_extensions
    if file_type not in allowed:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{file_type or 'unknown'}'. "
            f"Allowed types: {', '.join(sorted(allowed))}."
        )
    if not data:
        raise InvalidDocumentError("The uploaded file is empty.")
    size_limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > size_limit:
        raise DocumentTooLargeToIndexError(
            f"The upload exceeds the configured {settings.max_upload_mb} MB limit."
        )

    _validate_magic(file_type, data)

    if file_type == "pdf":
        pages = _extract_pdf(data, settings)
    elif file_type == "docx":
        pages = _extract_docx(data)
    else:
        pages = [ExtractedPage(page_number=1, text=_decode_text(data), method="text")]

    if not any(page.text.strip() for page in pages):
        hint = " If this is a scanned PDF, ensure OCR is enabled and installed." if file_type == "pdf" else ""
        raise NoExtractableTextError(
            "No extractable text was found in the uploaded document." + hint
        )

    extracted_chars = sum(len(page.text) for page in pages)
    if extracted_chars > settings.max_extracted_chars:
        raise DocumentTooLargeToIndexError(
            "Extracted text exceeds the configured limit. Split the document and retry."
        )

    logger.info(
        "Extracted %s pages from %s (%s, %s chars)",
        len(pages),
        safe_name,
        file_type,
        extracted_chars,
    )
    return ExtractedDocument(
        filename=safe_name,
        file_type=file_type,
        content_type=content_type or f"application/{file_type}",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        pages=pages,
    )
