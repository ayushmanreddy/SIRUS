import hashlib
import json
from app.config import Settings
from app.model_artifacts import verify_model_artifacts

def test_local_model_artifact_verifies(tmp_path):
    root = tmp_path / "models"; root.mkdir()
    model = root / "model.gguf"; model.write_bytes(b"local-model")
    (root / "manifest.json").write_text(json.dumps({"version": 1, "artifacts": [{"identifier": "demo", "role": "llm", "file": "model.gguf", "sha256": hashlib.sha256(b"local-model").hexdigest(), "runtime": "llama.cpp", "format": "GGUF", "min_system_ram_gb": 8, "min_vram_gb": 0}]}), encoding="utf-8")
    result = verify_model_artifacts(Settings(model_artifact_root=str(root), model_manifest_path=str(root / "manifest.json")))
    assert result.status == "ready"

def test_manifest_cannot_escape_models_directory(tmp_path):
    root = tmp_path / "models"; root.mkdir(); outside = tmp_path / "outside.gguf"; outside.write_bytes(b"x")
    (root / "manifest.json").write_text(json.dumps({"version": 1, "artifacts": [{"identifier": "unsafe", "role": "llm", "file": "../outside.gguf", "sha256": hashlib.sha256(b"x").hexdigest(), "runtime": "llama.cpp", "format": "GGUF", "min_system_ram_gb": 8, "min_vram_gb": 0}]}), encoding="utf-8")
    result = verify_model_artifacts(Settings(model_artifact_root=str(root), model_manifest_path=str(root / "manifest.json")))
    assert result.status == "unavailable"
