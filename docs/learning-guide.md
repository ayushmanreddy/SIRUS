# SIRUS V1 learning guide

This guide follows one question from the HTTP request to the final cited answer.
Read it with the code open. Each section names the files and functions involved
and explains the trade-offs you should be able to defend in an interview.

## 0. The mental model

SIRUS is not one model. It is a pipeline of small, replaceable components:

```
HTTP → validate → extract → normalize → chunk → embed → store
                                        ↑
question → validate → embed → search → threshold → prompt → LLM → answer + sources
```

Embeddings convert text into vectors so that semantically similar text is close
in vector space. Qdrant finds the closest stored vectors. The LLM never sees the
whole corpus; it sees only a handful of retrieved passages plus strict grounding
instructions. This is why RAG can answer from private documents without training
or fine-tuning.

Two invariants matter more than any individual file:

1. **The embedding used for ingestion and the embedding used for questions must
   come from the same model.** Otherwise distances are meaningless.
2. **Structured sources must come from storage, not from the model.** Citations
   are only trustworthy if they are copied from Qdrant payloads.

## 1. Configuration and dependency wiring

Files: `app/config.py`, `app/dependencies.py`, `app/main.py`.

`Settings` is a Pydantic settings model. Environment variables map to fields
case-insensitively (`EMBEDDING_MODEL` → `embedding_model`). Validators enforce:

- service URLs are absolute HTTP(S);
- overlap is smaller than chunk size;
- only supported file types are allowed;
- only SQLite is used for metadata in V1;
- `offline-demo` refuses public AI endpoints.

`create_app(settings, container)` is a factory. The container (`AppContainer`)
holds one configured instance of every adapter and service. Routes receive it via
`request.app.state.container` (`app/api/deps.py`), so business objects are not
created per request and tests can inject fakes.

**Why a container?** It makes dependencies explicit and avoids import-time
network calls. **Alternatives:** FastAPI `Depends` providers are also fine; a
dict/Dataclass container is simpler to read than a DI framework. **Trade-off:** a
process-wide container is shared state, but it is read-only after startup in V1.

## 2. Upload and validation

Files: `app/api/documents.py` (`ingest_document`, `_read_upload_limited`),
`app/documents/extraction.py`.

Starlette spools large uploads to disk, but SIRUS still reads in 1 MB chunks and
stops at `MAX_UPLOAD_MB`, so memory use is bounded. The filename is reduced to a
basename (`sanitize_filename`): `../../etc/passwd` becomes `passwd`. The filename
is metadata only; it is never used as a filesystem path.

`extract_document` checks:

- extension against `ALLOWED_FILE_TYPES`;
- non-empty data and size;
- magic bytes (`%PDF-`, DOCX ZIP header, binary NUL in text);
- page and extracted-character caps.

Failures raise domain errors in `app/errors.py` (`UnsupportedFileTypeError` → 415,
`InvalidDocumentError`/`NoExtractableTextError` → 422, etc.). `app/main.py`
registers one exception handler that converts them to safe JSON. This is the
difference between "the process crashed" and "the client receives an actionable
status code".

## 3. Text extraction

Files: `app/documents/extraction.py`, `app/ocr.py`.

PDF extraction is two-stage:

1. `page.get_text("text")` reads embedded digital text. This is exact, fast, and
   preserves reading order reasonably well.
2. If a page has fewer than `PDF_NATIVE_TEXT_MIN_CHARS`, render that page at
   `OCR_DPI` and run EasyOCR.

`extract_pdf_pages` opens the PDF once, tracks page numbers, and closes the
document in a `finally` block. OCR is CPU-heavy and synchronous, so
`IngestionService.ingest` runs the whole extraction in
`asyncio.to_thread(...)`; otherwise one scan could block every async request.

**Trade-offs:** always-OCR is simpler but far slower and can corrupt clean text.
Native-first is better for digital PDFs and reserves OCR for scans. OCR model
weights download on first use; that is documented as an offline prerequisite.

TXT tries `utf-8-sig`, `utf-8`, `cp1252`, then `latin-1`. DOCX reads paragraphs
and table rows; it has no reliable physical page concept, so page is `1`.

## 4. Chunking

File: `app/rag/chunking.py` (`normalize_text`, `chunk_text`).

Chunking exists because embedding models and LLM context windows are limited, and
because a focused passage retrieves better than a 40-page document. SIRUS chunks
by characters (default 800, overlap 120) for determinism across models.

`normalize_text` unifies line endings, removes control/form-feed characters,
collapses runs of spaces and blank lines, and keeps paragraph breaks. The splitter
prefers a paragraph break, then sentence punctuation, then whitespace in the last
40% of the target window. It never exceeds the target size and merges a tiny
trailing fragment into the previous chunk.

Each chunk carries `page_number`, `chunk_index`, `char_start`, `char_end`. These
become Qdrant payload fields and therefore citations. **Alternative:** token-based
splitting is more model-aware but requires a tokenizer and model-specific logic.
**Trade-off:** character chunks may split mid-sentence if no boundary exists, but
they are predictable and work with any embedding model.

## 5. Embeddings

File: `app/ai/embedding_ollama.py`.

`OllamaEmbeddingProvider` calls `POST /api/embed` with `{"model", "input"}`. It
validates:

- the response contains embeddings;
- each vector is non-empty and finite;
- lengths match any configured dimension;
- a batch returns exactly one vector per input;
- vectors in a batch have consistent dimensions.

Failures are split into "service unavailable" (HTTP/connection → 503) and
"invalid response" (bad vector/payload → 502). There is deliberately **no silent
fallback dimension**. A wrong default would create a collection with the wrong
vector size and corrupt future searches.

`embed_batch` is used at ingestion; `embed` is used for the question. Batching
reduces HTTP overhead and is capped by `EMBEDDING_BATCH_SIZE`.

## 6. Qdrant storage

Files: `app/ai/_vector_store.py`, `app/ai/vector_store_qdrant.py`,
`app/services/ingestion.py`.

The abstract interface (`VectorStoreInterface`) keeps ingestion/retrieval
independent of Qdrant REST details. The Qdrant adapter uses:

- `PUT /collections/{name}` to create a cosine collection;
- `GET /collections/{name}` for dimension/point count;
- `PUT /collections/{name}/points?wait=true` to upsert;
- `POST /collections/{name}/points/query` for similarity search;
- `POST .../points/delete` for cleanup.

Point IDs are UUIDv5 values derived from `document_id:page:chunk_index`. That makes
retries idempotent at the point level. The payload stores:

```json
{
  "id": "<uuid>",
  "chunk_id": "<uuid>",
  "document_id": "<uuid>",
  "source_document": "manual.pdf",
  "page_number": 3,
  "chunk_index": 2,
  "content": "the exact retrieved passage",
  "char_start": 812,
  "char_end": 1590,
  "embedding_model": "nomic-embed-text",
  "sha256": "..."
}
```

**Why separate SQLite from Qdrant?** Qdrant is good at vectors but listing
documents, duplicate detection, and collection/model bookkeeping are relational
concerns. SQLite gives cheap, transactional metadata. **Trade-off:** two stores
are not atomically committed; on failure SIRUS deletes partial points best-effort
and does not write a registry row.

Collection compatibility is checked in `_ensure_collection`:

1. probe the live embedding dimension;
2. read Qdrant's vector size;
3. compare with the registry's model/dimension state;
4. create only if missing; return 409 on mismatch; never delete data.

## 7. Idempotent ingestion and metadata

Files: `app/services/document_registry.py`, `app/services/ingestion.py`.

The registry table stores one row per indexed document: ID, sanitized filename,
content type, SHA-256, size, pages, chunks, embedding model/dimension, collection,
timestamp, and chunk summaries. A unique index on
`(sha256, embedding_model, collection_name)` defines duplicate identity.

The ingestion order is intentional:

1. extract and hash the file;
2. return the existing record immediately on duplicate (no re-embedding);
3. chunk and enforce `MAX_CHUNKS_PER_DOCUMENT`;
4. ensure the collection/dimension/model are compatible;
5. embed in batches;
6. upsert points;
7. only then write the registry row.

If step 6 fails, `delete_points` removes partial data. If step 7 races with a
concurrent duplicate upload, the unique index raises `IntegrityError`; SIRUS
deletes its own points and returns the existing document.

## 8. Retrieval

Files: `app/services/query.py`, `app/rag/retrieval.py`.

`QueryService.answer` first checks whether anything is indexed. If the registry is
empty, it inspects Qdrant's point count to support legacy collections; if there is
nothing, it returns the no-evidence answer **without calling the LLM**. This is
important: calling a model with no evidence invites hallucination.

`RetrievalService.retrieve` embeds the question with the same provider, searches
Qdrant with `top_k`, applies an optional `document_id` filter, drops chunks below
`RETRIEVAL_SCORE_THRESHOLD`, sorts by score, and renumbers citations. It returns
`RetrievedChunk` dataclasses — not HTTP objects — so retrieval is testable in
isolation.

**Threshold trade-off:** a high threshold improves precision (fewer irrelevant
chunks) but can miss paraphrases; zero maximizes recall but may feed noise to the
LLM. Tune it per embedding model.

## 9. Prompt construction and grounded generation

Files: `app/rag/prompts.py`, `app/llm_client.py`, `app/services/query.py`.

The system prompt tells the model to:

- use only the supplied evidence;
- never invent document contents or citations;
- cite evidence items as `[n]`;
- label inference;
- reply with the exact no-evidence sentence when evidence is insufficient.

Evidence blocks include the document name, page, chunk ID, and content. Prompt
construction is a pure function, so tests can assert the rules without an LLM.

`OllamaClient.generate` calls `/api/generate` with `stream: false`, optional
`system`, and optional `options.temperature`. HTTP failures become
`ModelUnavailableError` → 503. The answer is stripped and normalized. If it matches
the insufficiency markers, SIRUS replaces it with the canonical no-evidence answer
and sets `grounded: false`.

**Alternatives:** Ollama `/api/chat` with role messages; a larger model server
(vLLM); structured JSON output. `/api/generate` with a system prompt is the
smallest compatible choice. **Trade-off:** an LLM can still ignore instructions
or emit a wrong `[n]`; structured sources remain the source of truth.

## 10. Citations

Files: `app/rag/retrieval.py`, `app/api/documents.py`.

`RetrievedChunk` is built only from Qdrant search results. The API maps it to
`SourceReference` with `citation_id`, `document_id`, `document_name`, `chunk_id`,
`page`, `chunk_index`, `score`, and `text_preview`. The model does not generate
these fields. A user can:

1. use `document_id` with `GET /documents/{id}`;
2. use `chunk_id` to locate the exact Qdrant point;
3. compare `text_preview` and `score` with the answer.

If no evidence passes the threshold, sources are empty. SIRUS does not attach
irrelevant chunks just to appear grounded.

## 11. Health and readiness

Files: `app/health.py`, `app/api/system.py`.

Liveness means the process is up. Readiness means the system can actually answer:
SQLite schema exists, Qdrant is reachable, and **both configured models are
installed** (checked through Ollama `/api/tags`). A running Ollama container is not
enough. This distinction is what makes `/health/ready` trustworthy.

## 12. Testing strategy

Files: `tests/conftest.py`, unit tests, `tests/integration/`,
`scripts/e2e_smoke.py`.

- **Unit tests** cover chunk boundaries, extraction encodings/OCR fallback,
  embedding payload validation, Qdrant request shapes and payload parsing,
  registry duplicates, retrieval thresholds/filters, prompt rules, and query
  grounding.
- **API tests** use `TestClient` with a real `AppContainer` but fake embedding,
  Qdrant, and LLM adapters, plus real SQLite. They verify status codes, schemas,
  citations, and dependency failures.
- **Integration/smoke tests** run against real Ollama and Qdrant. Mocked tests
  prove wiring; only live tests prove the embedding → Qdrant → LLM flow.

A useful habit: for every failure mode (offline Qdrant, missing model, empty
question, corrupt PDF, dimension mismatch), ask "what status code and message
should a client receive?" Then find the test that proves it.

## 13. Interview questions you should be able to answer

1. Why is the same embedding model required for ingestion and query?
2. Why store text in the Qdrant payload instead of only vectors?
3. What exactly makes ingestion idempotent here?
4. Why use deterministic UUIDv5 point IDs? What would UUIDv4 break?
5. Why check the embedding dimension before creating a collection?
6. Why return 409 instead of silently recreating a mismatched collection?
7. Why is OCR a fallback rather than the default PDF path?
8. Why run extraction in a worker thread?
9. Why does the query path skip the LLM when no evidence passes the threshold?
10. How do structured citations differ from model-generated inline citations?
11. Why are duplicate uploads keyed by hash + model + collection?
12. What is the failure mode if the registry commit succeeds but the vector
    upsert is later manually deleted? (Stale metadata; documents list would show a
    document with no vectors.)
13. How would you add multiple knowledge bases? (Add a namespace/tenant field to
    collection naming and registry keys; inject a per-request container/config.)
14. How would you add authentication? (Middleware/dependency at the API layer;
    keep services independent of identity.)
15. What would you measure to evaluate this RAG system? (Retrieval recall@k,
    grounding/faithfulness, citation validity, refusal rate, latency per stage.)

## 14. Suggested exercises

1. Add `GET /documents/{id}/chunks` backed by a Qdrant scroll filter.
2. Add a `DELETE /documents/{id}` that removes both points and registry metadata
   transactionally with compensation.
3. Replace character chunking with a tokenizer and compare retrieval quality.
4. Add hybrid search (BM25 + vectors) and measure recall.
5. Validate that every `[n]` in the model answer exists in `sources`.
6. Add structured JSON output to the LLM and compare refusal reliability.
7. Add an offline evaluation set with known question/passage pairs.

If you can explain and extend this pipeline, you understand the core engineering
behind production RAG systems.
