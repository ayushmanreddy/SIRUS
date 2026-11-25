from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("sirus.ocr")

_reader: Optional["easyocr.Reader"] = None
_reader_lock = threading.Lock()
_reader_languages: tuple[str, ...] | None = None


class InvalidPDFError(Exception):
    """Raised when a PDF cannot be parsed by PyMuPDF."""


class PDFPageLimitError(InvalidPDFError):
    """Raised when a PDF exceeds the configured page limit."""


class OCRExtractionError(Exception):
    """Raised when OCR is required but cannot run."""


@dataclass(frozen=True)
class PDFPageText:
    page_number: int
    text: str
    method: str


def _get_reader(languages: list[str] | None = None) -> "easyocr.Reader":
    """Return a lazily initialized EasyOCR reader.

    EasyOCR downloads model weights on first use. For offline deployments,
    initialize the reader once with network access and keep the cache, or mount
    a pre-populated EasyOCR model directory.
    """
    global _reader, _reader_languages
    requested = tuple(languages or ["en"])
    with _reader_lock:
        if _reader is None or _reader_languages != requested:
            try:
                import easyocr
                import torch

                gpu = torch.cuda.is_available()
                logger.info("Initializing easyocr.Reader languages=%s gpu=%s", requested, gpu)
                _reader = easyocr.Reader(list(requested), gpu=gpu)
                _reader_languages = requested
            except Exception as error:  # noqa: BLE001 - normalized into a domain error
                logger.error("EasyOCR initialization failed: %s", error)
                raise OCRExtractionError("OCR is unavailable because EasyOCR could not be initialized.") from error
        return _reader


def render_pdf_pages(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Render each PDF page as PNG bytes (kept for the existing analysis flow)."""
    import fitz

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise InvalidPDFError("Failed to open PDF") from error
    try:
        pages: list[bytes] = []
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        for page_number in range(len(document)):
            pixmap = document[page_number].get_pixmap(matrix=matrix)
            pages.append(pixmap.tobytes("png"))
        return pages
    finally:
        document.close()


def ocr_png_bytes(png_bytes: bytes, languages: list[str] | None = None) -> str:
    """Run OCR on PNG bytes and return newline-joined text."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise OCRExtractionError("OpenCV/numpy are required for OCR.") from error

    image_np = np.frombuffer(png_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise OCRExtractionError("The rendered PDF page could not be decoded.")
    try:
        results = _get_reader(languages).readtext(image_bgr, detail=0)
    except OCRExtractionError:
        raise
    except Exception as error:  # noqa: BLE001
        logger.error("OCR failed: %s", error)
        raise OCRExtractionError("OCR failed while reading a document page.") from error
    return "\n".join(str(item) for item in results if str(item).strip())


def extract_pdf_pages(
    pdf_bytes: bytes,
    *,
    ocr_enabled: bool = True,
    languages: list[str] | None = None,
    dpi: int = 200,
    min_native_chars: int = 40,
    max_pages: int | None = None,
) -> list[PDFPageText]:
    """Extract per-page text from a PDF.

    Native embedded text is preferred because it is exact and fast. OCR is a
    fallback only for pages with little or no digital text (for example scans).
    This avoids paying OCR cost for normal PDFs and preserves page numbers.
    """
    import fitz

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise InvalidPDFError("Failed to open PDF") from error

    try:
        if max_pages is not None and len(document) > max_pages:
            raise PDFPageLimitError(f"PDF has more than the allowed {max_pages} pages.")

        pages: list[PDFPageText] = []
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        for index in range(len(document)):
            page_number = index + 1
            page = document[index]
            native_text = (page.get_text("text") or "").strip()
            if len(native_text) >= min_native_chars:
                pages.append(PDFPageText(page_number, native_text, "native"))
                continue

            if not ocr_enabled:
                pages.append(PDFPageText(page_number, native_text, "native-partial"))
                continue

            pixmap = page.get_pixmap(matrix=matrix)
            ocr_text = ocr_png_bytes(pixmap.tobytes("png"), languages)
            if ocr_text.strip():
                pages.append(PDFPageText(page_number, ocr_text, "ocr"))
            else:
                pages.append(PDFPageText(page_number, native_text, "native-partial"))

        return pages
    finally:
        document.close()


def extract_text(pdf_bytes: bytes) -> str:
    """Extract concatenated PDF text (native text first, OCR fallback).

    Retained for the document-analysis endpoint and existing callers.
    """
    pages = extract_pdf_pages(pdf_bytes)
    return "\n\n".join(page.text for page in pages)
