# Local runbook

## 1. Start the stack

```powershell
Copy-Item .env.example .env

# Docker path (Qdrant + Ollama + API)
docker compose up --build -d
docker compose --profile setup run --rm ollama-models
```

`ollama-models` pulls `LLM_MODEL` and `EMBEDDING_MODEL` from `.env`. Pulling is a
separate step; a running Ollama container does not include models.

Alternative if Ollama is installed on the host:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.1:8b
# .env: LLM/EMBEDDING_SERVICE_URL=http://host.docker.internal:11434
```

## 2. Verify readiness in order

```powershell
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/health/ready
Invoke-RestMethod http://localhost:8000/system/status
```

`/health/ready` is 200 only when:

- SQLite contains the registry and audit tables.
- Qdrant `/healthz` responds.
- Ollama lists the configured embedding model.
- Ollama lists the configured LLM model.

A 503 with a descriptive check is expected before models are pulled. Do not
report the system as ready merely because the container is running.

## 3. End-to-end smoke test

```powershell
python scripts/e2e_smoke.py --base-url http://localhost:8000
```

Or:

```powershell
$env:SIRUS_INTEGRATION="1"
python -m pytest tests/integration -v
```

The smoke script uploads a unique text document, asks a grounded question,
checks structured citations, and verifies an unrelated question is refused.

## 4. Check persistence

- `workbench-data` → `/app/data/workbench.db` (registry + audit).
- `qdrant-data` → `/qdrant/storage`.
- `ollama-data` → `/root/.ollama` (models).

Back up both named volumes before a destructive migration.

## Recovery

| Problem | Action |
| --- | --- |
| Readiness reports embedding model missing | `ollama pull <EMBEDDING_MODEL>` |
| Readiness reports LLM model missing | `ollama pull <LLM_MODEL>` |
| Qdrant unavailable | Inspect `docker compose logs qdrant`; do not delete volumes with data |
| API cannot reach Ollama | Fix `LLM_SERVICE_URL`; Compose uses `http://ollama:11434` |
| 409 collection incompatibility | Use a new `QDRANT_COLLECTION` and re-ingest, or follow the README rebuild procedure |
| OCR slow / unavailable on first scan | Run an OCR PDF once online to cache EasyOCR weights; check `OCR_ENABLED` |
| Duplicate upload skipped | Expected idempotency based on SHA-256 + model + collection |

## Embedding model change

Changing `EMBEDDING_MODEL` invalidates old vectors even if dimensions match.
SIRUS refuses to mix models in one collection. The safe procedure is:

1. Pull the new model.
2. Set the new `EMBEDDING_MODEL`.
3. Set a new `QDRANT_COLLECTION` name.
4. Restart and re-ingest documents.

Old data remains available for rollback. Deleting a collection while keeping the
SQLite registry rows would cause duplicates to be skipped, so never do a partial
destructive reset.

## Before declaring a demo complete

1. `/health/live` → 200.
2. `/health/ready` → 200 with all four checks `ready`.
3. Upload a real PDF/TXT/DOCX.
4. List documents and inspect the returned chunk metadata.
5. Ask a question answerable from the document.
6. Verify `grounded: true` and non-empty sources with real chunk IDs and scores.
7. Ask an unrelated question.
8. Verify `grounded: false` and the fixed no-evidence answer.
9. Confirm audit events exist in SQLite and contain no document/prompt text.
