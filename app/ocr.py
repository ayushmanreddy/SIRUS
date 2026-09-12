from __future__ import annotations

import logging

import easyocr
import numpy as np

logger = logging.getLogger("sovereign_workbench.ocr")

_reader = easyocr.Reader(["en"], gpu=True)  # loaded once at import time


class InvalidPDFError(Exception):
    """Raised when a PDF cannot be parsed by PyMuPDF."""


def render_pdf_pages(pdf_bytes: bytes) -> list[bytes]:
    """Render each page of a PDF as PNG bytes using PyMuPDF (fitz).

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        List of PNG byte chunks, one per page.

    Raises:
        ImportError: If PyMuPDF (fitz) is not installed.
        InvalidPDFError: If the PDF cannot be opened.
    """
    import fitz

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise InvalidPDFError("Failed to open PDF") from error
    pages: list[bytes] = []
    for page_num in range(len(document)):
        page = document[page_num]
        pixmap = page.get_pixmap()
        pages.append(pixmap.tobytes("png"))
    document.close()
    return pages


def extract_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF pages using easyocr.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        Concatenated text from all pages, with page breaks indicated by double newlines.

    Raises:
        ImportError: If easyocr is not installed.
        InvalidPDFError: If the PDF cannot be opened.
    """
    import fitz

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise InvalidPDFError("Failed to open PDF") from error

    all_text: list[str] = []
    for page_num in range(len(document)):
        page = document[page_num]
        pixmap = page.get_pixmap()
        image_bytes = pixmap.tobytes("png")
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        import cv2
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        results = _reader.readtext(image_bgr, detail=0)
        page_text = "\n".join(results)
        all_text.append(page_text)
    document.close()
    return "\n\n".join(all_text)