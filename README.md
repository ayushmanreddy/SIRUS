# Sovereign On-Premise Agentic AI Workbench

An SIH internal-competition MVP for confidential-document AI workflows that keeps model-facing services inside the local deployment boundary.

## What works in this foundation

- FastAPI service with liveness, readiness, version, and presenter-safe system status endpoints
- Typed environment configuration with `development`, `demo`, and `offline-demo` profiles
- Local SQLite audit trail and structured JSON logging with secret/content redaction
- Docker Compose services for the API, Qdrant vector store, and local-model/embedding service slots
- A local-only endpoint policy: `offline-demo` refuses public AI service URLs
- **Document ingestion RAG pipeline**: `/documents/ingest` (PDF upload → OCR → chunking → embedding → Qdrant store) and `/documents/query` (similarity search → LLM answer with source citations)
- Fixed-size character chunking with overlap preserving page numbers and chunk indices
- Ollama-backed embedding provider via `/api/embeddings`
- Qdrant client with PUT/points?wait=true for insert and filter-based search

## Quick start

1. Copy `.env.example` to `.env` and keep it untracked.
2. Run `docker compose up --build`.
3. Open `http://localhost:8000/docs` or run the commands below.

```powershell
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/health/ready
Invoke-RestMethod http://localhost:8000/system/status
Invoke-RestMethod -Method Post -Uri http://localhost:8000/documents/ingest -Body @{file=@"/path/to/doc.pdf"} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/documents/query -Body @{question="What is the key finding?"} | ConvertTo-Json
```

For the foundation-only stack, readiness will report model and embedding services as unavailable until local runtimes are configured. Qdrant is included in Compose and should become ready automatically.

Epic 2 adds `GET /system/preflight`: it reports safe hardware capability details and verifies that approved model files exist locally with matching SHA-256 checksums. It never downloads model weights or reports local filesystem paths.

## Offline demonstration checklist

1. Place approved model artifacts on the demo machine before disconnecting it.
2. Set `APP_PROFILE=offline-demo` and point all AI service URLs to loopback/internal addresses.
3. Start the stack and check `/system/status`.
4. Disable external networking and run the health checks again.
5. Record the result. The status endpoint intentionally avoids exposing secrets and file paths.

See [docs/architecture.md](docs/architecture.md) for boundaries and [docs/runbook.md](docs/runbook.md) for recovery steps.