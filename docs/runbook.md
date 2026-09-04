# Local runbook

## Normal startup

1. Copy `.env.example` to `.env`.
2. For an offline proof, set `APP_PROFILE=offline-demo` and use only local/private endpoints.
3. Start `docker compose up --build`.
4. Verify `/health/live`, then `/health/ready`, then `/system/status`.

## Expected foundation state

Qdrant should show ready. LLM and embeddings will remain unavailable until the selected local runtimes from Sovereign Infrastructure are started. This is expected; do not describe the platform as fully ready before these services are connected.

## Recovery

- **Database unavailable:** retain the `workbench-data` volume, inspect container logs, then restart the API.
- **Vector store unavailable:** check the `qdrant` container and its persistent volume; do not delete a volume containing real demo data without an approved backup/reset plan.
- **LLM/embedding unavailable:** confirm the local runtime is started, its address is configured, and model artifacts exist locally. Do not substitute a hosted endpoint in `offline-demo`.
