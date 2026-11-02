# Architecture boundary

```text
Browser / presenter
       |
       v
FastAPI platform service ----> Local SQLite audit store
         |                  \
         |                   ---> Local Qdrant vector store (VectorStoreInterface)
         |                  /
         |                 /
         |                /
         v               /
  ModelRegistry (active model, model metadata, status)
  HardwareSnapshot (CPU, RAM, GPU detection)
  MonitoringSnapshot (CPU, RAM, GPU, inference latency metrics)
```

Core abstractions:

- `ModelProvider` (app.ai) — interface with load(), unload(), health(), generate(), stream()
- `OllamaClient` (app.llm_client) — implements ModelProvider for local Ollama generation
- `EmbeddingProvider` (app.ai._embedding) — interface with embed(), embed_batch(), health()
- `OllamaEmbeddingProvider` (app.ai.embedding_ollama) — implements EmbeddingProvider via POST /api/embeddings, model default "nomic-embed-text"
- `VectorStoreInterface` (app.ai._vector_store) — interface with create_collection(), insert(), similarity_search(), delete()
- `QdrantClient` (app.ai.vector_store_qdrant) — implements VectorStoreInterface using Qdrant REST API (PUT /collections/{name} for create, PUT /collections/{name}/points?wait=true for insert with UUID point IDs, filter-based search)
- `Chunking` (app.rag.chunking) — fixed-size character chunks with overlap, preserving page number and chunk_index
- `AuditStore` (app.audit) — local SQLite event store with redaction
- `Settings` (app.config) — typed environment config with offline-demo URL validation

`offline-demo` validates that every configured model-facing endpoint is loopback/private IP space or one of the explicitly named internal Compose services (llm, embeddings, qdrant). This is application-level protection, not a replacement for host firewall controls or enterprise network segmentation.

The platform records operational events only. Raw prompts, document text, authorization headers, and secret-bearing values are redacted before they enter the foundation audit trail.

Model artifact verification (SHA-256 checksums) ensures only approved local models are used; no runtime downloads occur.

## API endpoints

- `POST /documents/ingest` — Upload PDF, extract text via OCR, chunk, embed with Ollama, store in Qdrant "documents" collection. Returns `{document_id, chunks}`.
- `POST /documents/query` — Embed question, similarity search in Qdrant, build prompt from context, generate answer via local LLM. Returns `{answer, sources}` where sources include `{document_id, page, chunk_index, score, text_preview}`.
- `POST /agent/execute_code` — Demo-level unauthenticated code execution endpoint.