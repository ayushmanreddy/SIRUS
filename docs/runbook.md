# Local runbook

## Normal startup

1. Copy `.env.example` to `.env`.
2. For an offline proof, set `APP_PROFILE=offline-demo` and use only local/private endpoints.
3. Start `docker compose up --build`.
4. Verify `/health/live`, then `/health/ready`, then `/system/status`.
5. Call `GET /system/preflight` — verify all model artifacts are ready before proceeding.

## Expected foundation state

- Qdrant should show ready. LLM and embedding services will remain unavailable until local runtimes (Ollama with approved model, local embedding model) are started and model artifacts verify successfully. This is expected; do not describe the platform as fully ready before these services are connected.
- `GET /system/status` will show profile, uptime, service hosts, and checks.
- `GET /system/preflight` will show `status: "ready"` if all SHA-256 checksums match.

## Recovery

- **Database unavailable:** retain the `workbench-data` volume, inspect container logs, then restart the API.
- **Vector store unavailable:** check the `qdrant` container and its persistent volume; do not delete a volume containing real demo data without an approved backup/reset plan.
- **LLM/embedding unavailable:** confirm the local Ollama runtime is started and its address is configured (`LLM_SERVICE_URL`), model artifacts exist locally with matching checksums, and the model is loaded. Do not substitute a hosted endpoint in `offline-demo`.
- **Embedding unavailable:** confirm the local embedding runtime is started and its address is configured (`EMBEDDING_SERVICE_URL`), or that the `LocalEmbeddingProvider` fallback is active.
- **Model artifacts unavailable:** verify the `models/manifest.json` file exists and all SHA-256 checksums match. Place approved model files in the `models/` directory.

## Offline demonstration checklist

1. Place approved model artifacts on the demo machine before disconnecting it.
2. Set `APP_PROFILE=offline-demo` and point all AI service URLs to loopback/internal addresses (e.g., `LLM_SERVICE_URL=http://127.0.0.1:8081`, `EMBEDDING_SERVICE_URL=http://localhost:8082`, `VECTOR_STORE_URL=http://192.168.1.20:6333`).
3. Start the stack: `docker compose up --build`.
4. Verify `/health/live` returns `{"status":"live"}`.
5. Verify `/health/ready` — some checks may report `degraded` if local runtimes aren't fully connected yet; this is expected.
6. Call `GET /system/status` — verify all service hosts are reachable.
7. Call `GET /system/preflight` — artifacts should report `status: "ready"` if all checksums match.
8. Disable external networking and run the health checks again.
9. Record the result. The status endpoint intentionally avoids exposing secrets and file paths.
10. Verify that no external AI APIs are invoked (check container network traffic if needed).

Model weights, credentials, source download URLs, prompts, and confidential documents stay out of Git.