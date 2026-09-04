# Architecture boundary

```text
Browser / presenter
       |
       v
FastAPI platform service ----> Local SQLite audit store
       |                 \
       |                  ---> Local Qdrant vector store
       |                 \
       |                  ---> Local LLM runtime (next integration)
       |                 \
       |                  ---> Local embedding runtime (next integration)
```

`offline-demo` validates that every configured model-facing endpoint is loopback, private IP space, or one of the explicitly named internal Compose services. This is application-level protection, not a replacement for host firewall controls or enterprise network segmentation.

The platform records operational events only. Raw prompts, document text, authorization headers, and secret-bearing values are redacted before they enter the foundation audit trail.
