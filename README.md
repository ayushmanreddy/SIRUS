# SIRUS — Sovereign Local AI Workbench

SIRUS is a local-first FastAPI service for document analysis and retrieval-augmented question answering (RAG). It can connect to local model and embedding services, stores vectors in Qdrant, and keeps operational audit data in SQLite.

> This is an MVP. The API has no authentication. Keep it on a trusted local network, and do not expose the demo code-execution endpoint to untrusted users.

## What it includes

- PDF ingestion: OCR, overlapping text chunks, embeddings, and Qdrant storage.
- Document questions answered from retrieved context, with page and chunk citations.
- PDF analysis and summarization through a local language model.
- Ollama-compatible clients for text generation and embeddings.
- Health, status, and preflight endpoints.
- SQLite operational audit events and structured logs.
- An `offline-demo` profile that rejects configured public service URLs.
- Local model artifact manifest and SHA-256 verification.

## Requirements

For the Docker setup:

- Docker Desktop with Docker Compose.
- A local model service that provides an Ollama-compatible API, such as Ollama, for generation and embeddings.

For running the API directly on Windows, use Python 3.11 or newer. Qdrant and a local model service must still be running separately.

## Quick start with Docker

1. Clone the repository and open a terminal in the `SIRUS` folder.
2. Create your local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Edit `.env`. If Ollama is running on your Windows host, set:

   ```dotenv
   LLM_SERVICE_URL=http://host.docker.internal:11434
   EMBEDDING_SERVICE_URL=http://host.docker.internal:11434
   VECTOR_STORE_URL=http://qdrant:6333
   ```

   The default models are `llama3.1:8b` for generation and `nomic-embed-text` for embeddings. Make sure those models are available in your local runtime. Change the model settings in `app/config.py` if your setup uses different names.

4. Build and start the API and Qdrant:

   ```powershell
   docker compose up --build
   ```

5. Open the interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs). Use the `documents` endpoints to upload a PDF and ask a question.

Stop the services with `Ctrl+C`. The Compose file stores SQLite and Qdrant data in named volumes so it persists across restarts.

## Try the API

Check that the API is running:

```powershell
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/system/status
```

Upload a PDF (replace the path with your own file):

```powershell
curl.exe -X POST "http://localhost:8000/documents/ingest" -F "file=@C:\path\to\document.pdf"
```

Ask a question about ingested documents:

```powershell
$body = @{ question = "What is the key finding?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/documents/query" -Method Post -ContentType "application/json" -Body $body
```

The ingestion endpoint accepts PDFs up to 10 MB and 100 pages by default. These limits can be changed with `MAX_UPLOAD_MB` and `MAX_PAGE_COUNT`.

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Liveness check |
| `GET` | `/health/ready` | Readiness checks for the database and configured services |
| `GET` | `/system/version` | Application name and version |
| `GET` | `/system/status` | Profile, service hosts, readiness, and model artifact status |
| `GET` | `/system/preflight` | Hardware summary and local model artifact verification |
| `POST` | `/documents/ingest` | OCR, chunk, embed, and store a PDF |
| `POST` | `/documents/query` | Retrieve relevant chunks and answer a question with citations |
| `POST` | `/documents/analyze` | Extract PDF text and generate a summary |
| `POST` | `/agent/generate` | Generate a response for a prompt |
| `POST` | `/agent/execute_code` | Run demo code in the service sandbox |

The interactive docs at `/docs` show request and response schemas. The code-execution endpoint is an unauthenticated demo feature; do not expose it to untrusted users.

The current test suite uses mocked dependencies for RAG route tests. Passing unit tests does not verify the complete PDF to local model to Qdrant flow against live services; validate that flow in your environment before relying on it.

## Configuration

Settings are read from environment variables and, when running locally, from `.env`. Common settings:

| Variable | Default | Description |
| --- | --- | --- |
| `APP_PROFILE` | `development` | `development`, `demo`, or `offline-demo` |
| `LLM_SERVICE_URL` | `http://localhost:11434` | Local text-generation service base URL |
| `EMBEDDING_SERVICE_URL` | `http://localhost:11434` | Local embedding service base URL |
| `VECTOR_STORE_URL` | `http://localhost:6333` | Qdrant base URL |
| `MAX_UPLOAD_MB` | `10` | Maximum PDF size accepted by document ingestion |
| `MAX_PAGE_COUNT` | `100` | Maximum PDF page count accepted by document ingestion |

In `offline-demo`, all configured model and vector-store URLs must use loopback, private-network, or explicitly supported internal service hosts. This application check complements network controls; it does not replace firewall rules.

Keep `.env`, model weights, and confidential documents out of Git. The `models/` directory is mounted read-only in Docker for local artifacts.

## Local development

Create a virtual environment, install dependencies, and run the tests:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

To run the API outside Docker:

```powershell
python -m uvicorn app.main:app --reload
```

Configure service URLs for the host environment in `.env`. Qdrant is not started by this command; start it separately or use Docker Compose.

## Project layout

- `app/main.py` — FastAPI routes and application setup
- `app/ai/` — embedding and vector-store clients
- `app/rag/` — text chunking
- `app/ocr.py` — PDF rendering and OCR helpers
- `configs/` — application configuration files
- `docs/` — architecture notes and local runbook
- `tests/` — API and component tests

## Documentation

- [Architecture](docs/architecture.md)
- [Local runbook](docs/runbook.md)
- [Epic 2 local models](docs/epic-2-local-models.md)

