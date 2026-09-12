"""OCR pipeline test - verifies PDF→image→text pipeline is wired correctly.

This test builds a minimal single-page PDF in-memory using fitz.open() + page.insert_text()
with known text, run extract_text() on it, and assert the known text substring appears
in the result. Does NOT test OCR accuracy on real scanned images.
"""
import io

import fitz
import numpy as np
import pytest

from app.ocr import extract_text


def test_ocr_pdf_pipeline():
    """Build a minimal single-page PDF with known text and verify extract_text recovers it."""
    # Create a single-page PDF with known text
    # Using "Sovereign AI Workbench" as test text - OCR may drop certain characters
    doc = fitz.open()
    page = doc.new_page()
    # Use text that OCR handles well; place it clearly
    page.insert_text((72, 72), "Sovereign AI Workbench", fontsize=18)

    # Save PDF to bytes
    pdf_bytes = doc.write()
    doc.close()

    # Extract text using the pipeline
    text = extract_text(pdf_bytes)

    # The known text substring should appear in the extracted result.
    # OCR may drop/change some characters; we check for a stable substring.
    assert "Sovereign" in text