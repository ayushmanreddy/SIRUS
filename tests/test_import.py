"""test_import.py: import app.main succeeds without EasyOCR weights."""

def test_import_app_main():
    """Import app.main without triggering EasyOCR Reader at import time."""
    import app.main  # noqa: F401