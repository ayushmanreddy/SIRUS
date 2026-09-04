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
