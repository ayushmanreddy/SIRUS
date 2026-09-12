# Epic 2: local model infrastructure

This increment adds a versioned local model manifest, checksum verification, path-traversal protection, safe CPU/GPU discovery, model abstraction layer, Ollama integration, model management, resource monitoring, local embeddings, and vector store abstraction.

## Implemented features

1. **ModelProvider abstraction** (`app.ai.ModelProvider`) — interface defining `load()`, `unload()`, `health()`, `generate()`, `stream()`. Allows swapping model providers without rewriting the application.

2. **Ollama integration** (`app.ai.ollama_client.OllamaClient`) — local inference backend communicating with Ollama's `/api/generate` and `/api/tags` endpoints. Includes:
   - `generate()` — non-streaming text generation
   - `stream()` — streaming text generation
   - `health()` — service health and available models check
   - `load()` / `unload()` — model lifecycle management
   - Timeout and error handling with `ModelUnavailableError`

3. **Model management** (`app.ai.model_management.ModelRegistry`) — registers available models, tracks active model, provides status (`is_loaded`, `is_healthy`, `latency_ms`, `requests_served`), supports setting active model and listing registered models.

4. **Model artifact verification** (`app.model_artifacts`) — SHA-256 checksum verification of locally transferred model artifacts. No downloads are performed at runtime. Path traversal protection ensures models cannot escape the configured artifact root directory.

5. **Hardware detection** (`app.hardware.detect_hardware`) — detects:
   - CPU: core count, model name (via /proc/cpuinfo), architecture
   - RAM: total system memory in MB
   - GPU: NVIDIA (via nvidia-smi) and AMD (via rocm-smi) with memory info
   - Falls back to CPU-only operation when no supported GPU is available

6. **Resource monitoring** (`app.monitor`) — snapshot-based infrastructure for:
   - CPU utilization (percentage, core frequencies, temperature)
   - RAM utilization (total, used, free, swap)
   - GPU utilization (percentage, memory, temperature, driver version)
   - Model inference performance (latency, throughput, error count)
   - `MonitoringSnapshot.to_dict()` / `from_dict()` for JSON serialization

7. **Local embedding infrastructure** (`app.ai.embedding_local.LocalEmbeddingProvider`) — implements `EmbeddingProvider` interface:
   - `embed()` — single text embedding (768-dimensional hash-based fallback)
   - `embed_batch()` — batch embedding generation
   - `health()`, `get_dimension()`, `get_model_name()`
   - Deterministic SHA-256 based embedding for development/demo when no model is available

8. **Vector store abstraction** (`app.ai._vector_store.VectorStoreInterface`) — interface for vector database operations:
   - `create_collection()`, `insert()`, `similarity_search()`, `delete()`, `delete_collection()`, `list_collections()`, `get_collection_info()`
   - `VectorStoreSearchResult` — search results with metadata and similarity scores
   - `VectorStoreMetadata` — metadata associated with vectors (id, content, source_document, page_number, chunk_index)

9. **Qdrant vector store** (`app.ai.vector_store_qdrant.QdrantClient`) — implements `VectorStoreInterface` for Qdrant:
   - Creation and management of collections
   - Vector insertion with metadata
   - Similarity search with metadata filtering
   - Vector and collection deletion
   - Health checks and collection information

10. **Offline-first enforcement** (`app.config`) —:
    - `AppProfile.OFFLINE_DEMO` blocks external AI service URLs
    - `block_external_ai_endpoints_offline()` model validator rejects non-loopback/private URLs
    - `is_local_or_private_url()` — validates URLs are loopback or private IP space
    - `GET /system/preflight` — reports artifact readiness without exposing paths or secrets
    - Model artifacts verified via SHA-256 checksums before use

## Model manifest workflow

1. Place approved model weights in the `models/` directory (untracked)
2. Copy `configs/model-manifest.example.json` to `models/manifest.json`
3. Replace example filenames and checksums with locally approved values
4. The manifest format follows `ModelManifest` -> `ModelArtifact` Pydantic models
5. Runtime verification via `verify_model_artifacts(settings)` — returns `ArtifactVerification(status="ready")` or `ArtifactVerification(status="unavailable")`

## Offline demonstration checklist

1. Place approved model artifacts on the demo machine before disconnecting
2. Set `APP_PROFILE=offline-demo` and point all AI service URLs to loopback/internal addresses
3. Start the stack and check `/health/live`, then `/health/ready`, then `/system/status`
4. Disable external networking and run health checks again
5. Call `/system/preflight` — artifacts are ready only if every checksum matches
6. Record the result. The status endpoint intentionally avoids exposing secrets and file paths.

Model weights, credentials, source download URLs, prompts, and confidential documents stay out of Git.