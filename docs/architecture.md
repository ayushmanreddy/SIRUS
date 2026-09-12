# Architecture boundary

```text
Browser / presenter
       |
       v
FastAPI platform service ----> Local SQLite audit store
        |                  \
        |                   ---> Local Qdrant vector store (VectorStoreInterface)
        |                  \
        |                   ---> Local LLM runtime (OllamaClient implements ModelProvider)
        |                  \
        |                   ---> Local embedding runtime (LocalEmbeddingProvider implements EmbeddingProvider)
        |
        v
  ModelRegistry (active model, model metadata, status)
  HardwareSnapshot (CPU, RAM, GPU detection)
  MonitoringSnapshot (CPU, RAM, GPU, inference latency metrics)
```

Core abstractions:

- `ModelProvider` (app.ai) — interface with load(), unload(), health(), generate(), stream()
- `OllamaClient` (app.ai.ollama_client) — implements ModelProvider for local Ollama
- `EmbeddingProvider` (app.ai._embedding) — interface with embed(), embed_batch(), health()
- `VectorStoreInterface` (app.ai._vector_store) — interface with create_collection(), insert(), similarity_search(), delete()
- `ModelRegistry` (app.ai.model_management) — manages available models, active model, model status
- `HardwareSnapshot` (app.hardware) — CPU, RAM, GPU detection with safe_dict()
- `MonitoringSnapshot` (app.monitor) — resource utilization metrics with JSON serialization

`offline-demo` validates that every configured model-facing endpoint is loopback/private IP space or one of the explicitly named internal Compose services (llm, embeddings, qdrant). This is application-level protection, not a replacement for host firewall controls or enterprise network segmentation.

The platform records operational events only. Raw prompts, document text, authorization headers, and secret-bearing values are redacted before they enter the foundation audit trail.

Model artifact verification (SHA-256 checksums) ensures only approved local models are used; no runtime downloads occur.