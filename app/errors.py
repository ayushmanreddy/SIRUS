"""Application-level exceptions.

Routes translate these into safe HTTP responses. Messages in this module are
deliberately user-facing: they must never contain credentials, host paths, or
upstream stack traces. Low-level details are logged by the component that
raises the error.
"""
from __future__ import annotations


class SirusError(Exception):
    """Base class for expected, user-facing SIRUS failures."""

    status_code: int = 500
    default_detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


class FileTooLargeError(SirusError):
    status_code = 413
    default_detail = "The uploaded file is larger than the configured limit."


class UnsupportedFileTypeError(SirusError):
    status_code = 415
    default_detail = "The uploaded file type is not supported."


class InvalidDocumentError(SirusError):
    status_code = 422
    default_detail = "The uploaded document could not be read."


class NoExtractableTextError(SirusError):
    status_code = 422
    default_detail = "No extractable text was found in the uploaded document."


class DocumentTooLargeToIndexError(SirusError):
    status_code = 422
    default_detail = "The document is too large to index."


class DocumentNotFoundError(SirusError):
    status_code = 404
    default_detail = "The requested document was not found."


class CollectionNotFoundError(SirusError):
    status_code = 404
    default_detail = "The vector collection does not exist yet."


class UpstreamResponseError(SirusError):
    status_code = 502
    default_detail = "An upstream service returned an invalid response."


class CollectionCompatibilityError(SirusError):
    status_code = 409
    default_detail = (
        "The configured embedding model is incompatible with the existing vector "
        "collection. Rebuild the collection after changing embedding models."
    )


class DependencyUnavailableError(SirusError):
    status_code = 503
    default_detail = "A required dependency is unavailable."


class EmbeddingUnavailableError(DependencyUnavailableError):
    default_detail = "The embedding service is unavailable."


class EmbeddingResponseError(SirusError):
    status_code = 502
    default_detail = "The embedding service returned an invalid response."


class VectorStoreUnavailableError(DependencyUnavailableError):
    default_detail = "The vector store is unavailable."


class LLMUnavailableError(DependencyUnavailableError):
    default_detail = "The language model service is unavailable."


class OCRUnavailableError(DependencyUnavailableError):
    default_detail = "OCR is required for this document but is unavailable."
