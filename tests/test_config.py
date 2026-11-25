import pytest
from pydantic import ValidationError

from app.config import Settings


def test_offline_demo_accepts_loopback_services():
    settings = Settings(
        app_profile="offline-demo",
        llm_service_url="http://127.0.0.1:8081",
        embedding_service_url="http://localhost:8082",
        vector_store_url="http://192.168.1.20:6333",
    )
    assert settings.app_profile == "offline-demo"


def test_offline_demo_rejects_public_ai_endpoint():
    with pytest.raises(ValidationError):
        Settings(app_profile="offline-demo", llm_service_url="https://api.example.com/v1")


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_allowed_file_types_are_validated_and_normalized():
    settings = Settings(allowed_file_types=".TXT, pdf")
    assert settings.allowed_extensions == frozenset({"pdf", "txt"})
    with pytest.raises(ValidationError):
        Settings(allowed_file_types="pdf,exe")


def test_retrieval_settings_and_models_are_configurable():
    settings = Settings(
        llm_model="qwen2.5:0.5b",
        embedding_model="nomic-embed-text",
        retrieval_top_k=7,
        retrieval_score_threshold=0.4,
        qdrant_collection="private_docs",
    )
    assert settings.answer_model == "qwen2.5:0.5b"
    assert settings.retrieval_top_k == 7
    assert settings.qdrant_collection == "private_docs"

