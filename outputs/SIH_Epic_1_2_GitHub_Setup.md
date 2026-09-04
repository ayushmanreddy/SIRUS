# SIH Sovereign On-Premise Agentic AI Workbench

GitHub-ready delivery plan for the internal SIH selection. This plan deliberately optimizes for a credible, working local MVP rather than a broad, unfinished enterprise platform.

## Project conventions

### Milestones

| Milestone | Purpose | Exit condition |
|---|---|---|
| **M1 — Foundation & Demo Skeleton** | Establish a reproducible local development baseline. | A fresh developer machine can start the skeleton service, run checks, and produce diagnosable logs. |
| **M2 — Sovereign Infrastructure & Offline Proof** | Run all model-facing infrastructure locally and demonstrate no runtime cloud dependency. | The local services start from locally available artifacts, report health/resources, and complete the offline proof. |

### Labels to create once

`epic: foundation`, `epic: sovereign-infrastructure`, `type: feature`, `type: chore`, `type: test`, `type: documentation`, `area: platform`, `area: backend`, `area: ml`, `area: devops`, `area: security`, `priority: P0`, `priority: P1`, `blocked`, `demo-critical`.

Priority rule: **P0** must be ready for judging; **P1** is included only after the P0 path is stable. Do not assign owners until the actual team membership is confirmed.

---

# Epic 1 — Foundation & Demo Skeleton

**Create as parent issue**

**Title:** `[Epic 1] Foundation & Demo Skeleton`

**Milestone:** `M1 — Foundation & Demo Skeleton`  
**Labels:** `epic: foundation`, `priority: P0`, `demo-critical`

## Goal

Provide a repeatable, observable development foundation on which the local model, document pipeline, RAG, and UI can be added without a late-demo integration scramble.

## MVP boundary

This epic delivers the application shell, configuration, persistence baseline, developer environment, automated checks, and demo operating notes. It does **not** claim to deliver full RAG, authentication, production Kubernetes, or a public-cloud deployment.

## Epic acceptance criteria

- A new contributor can follow one documented path to start the platform locally.
- The API process exposes liveness and readiness endpoints and returns a request identifier with failures.
- Configuration uses environment variables/profile files; no credentials or confidential documents are committed.
- A basic database migration and a structured audit/event record work locally.
- The automated quality checks run on each pull request.
- A teammate can follow the demo runbook without verbal setup guidance.

## Child issues

### FND-01 — Establish repository governance and protected contribution flow

**Title:** `FND-01: Establish repository governance and contribution flow`  
**Labels:** `epic: foundation`, `type: documentation`, `area: platform`, `priority: P0`

**Why:** A clear branch/PR flow prevents demo-critical work from being accidentally overwritten.

**Work:**

- Add `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `.gitignore`, and a pull-request template.
- Define `main` as the demo-ready branch and document short-lived `feature/<area>-<summary>` branches.
- Add a lightweight issue template with goal, scope, dependencies, acceptance criteria, and test evidence.
- Enable (or document enabling) review-before-merge and required CI checks where repository permissions allow it.
- Include a strict rule never to commit `.env`, model weights, private datasets, or confidential SIH material.

**Acceptance criteria:**

- [ ] New work has a documented branch, issue, PR, and review path.
- [ ] The PR template asks for test evidence and linked issue(s).
- [ ] Secret/model/data exclusions are present in `.gitignore`.
- [ ] The repository landing page explains the SIH MVP boundary.

**Dependencies:** None.

### FND-02 — Create reproducible local development stack

**Title:** `FND-02: Create reproducible local development stack`  
**Labels:** `epic: foundation`, `type: feature`, `area: devops`, `priority: P0`, `demo-critical`

**Work:**

- Provide a Docker Compose-based local stack for the API, database, and later local AI services through profiles/placeholders.
- Pin runtime versions and publish a sample environment file containing no secrets.
- Add start, stop, rebuild, and clean-state instructions.
- Include volume paths for persistent local data, logs, and model artifacts; exclude their contents from Git.
- Make failure output actionable when a required port, volume, or environment variable is missing.

**Acceptance criteria:**

- [ ] A clean clone can start the base services using only documented prerequisites.
- [ ] API and database containers have named, persistent local volumes.
- [ ] `sample.env` contains every required variable with safe example values.
- [ ] Startup fails with a clear message for a missing mandatory variable.

**Dependencies:** FND-01.

### FND-03 — Implement configuration profiles and secret-safe validation

**Title:** `FND-03: Implement configuration profiles and secret-safe validation`  
**Labels:** `epic: foundation`, `type: feature`, `area: backend`, `area: security`, `priority: P0`

**Work:**

- Define typed configuration for `development`, `demo`, and `offline-demo` profiles.
- Validate required settings at process startup, including local service URLs, data locations, and model artifact roots.
- Keep secrets in environment variables or an untracked local file; redact them from logs and error responses.
- Publish a configuration reference listing purpose, allowed values, and safe default for every setting.

**Acceptance criteria:**

- [ ] An invalid configuration prevents startup and identifies the setting, without exposing any secret value.
- [ ] `offline-demo` has no cloud endpoint/configuration requirement.
- [ ] The configuration reference matches the settings accepted by the application.

**Dependencies:** FND-02.

### FND-04 — Build API skeleton with liveness, readiness, and version endpoints

**Title:** `FND-04: Build API skeleton with operational endpoints`  
**Labels:** `epic: foundation`, `type: feature`, `area: backend`, `priority: P0`, `demo-critical`

**Work:**

- Create the service entry point and versioned API route namespace.
- Implement `/health/live`, `/health/ready`, and `/system/version`.
- Make readiness reflect dependencies registered by the application, rather than always returning healthy.
- Return stable JSON error envelopes and propagate a request/correlation identifier.

**Acceptance criteria:**

- [ ] Liveness succeeds when the process is running.
- [ ] Readiness fails when a required dependency is unavailable and names only the failing component class, not sensitive details.
- [ ] Version output identifies the build/version without leaking configuration.
- [ ] Errors include a request identifier suitable for log lookup.

**Dependencies:** FND-03.

### FND-05 — Add baseline persistence and schema migrations

**Title:** `FND-05: Add baseline persistence and schema migrations`  
**Labels:** `epic: foundation`, `type: feature`, `area: backend`, `priority: P0`

**Work:**

- Select one local relational database supported by the demo stack.
- Add versioned migrations and a documented empty-database bootstrap process.
- Create minimal tables for system events and later document/model metadata; do not store document content or prompts in this epic.
- Ensure migration failures are visible in readiness/log output.

**Acceptance criteria:**

- [ ] A fresh local database reaches the current schema through one documented command.
- [ ] Re-running migrations is safe.
- [ ] A versioned migration history is committed and reviewed.
- [ ] The service readiness check detects an unreachable or unmigrated database.

**Dependencies:** FND-02, FND-04.

### FND-06 — Implement structured logs and minimal audit event trail

**Title:** `FND-06: Implement structured logs and minimal audit event trail`  
**Labels:** `epic: foundation`, `type: feature`, `area: backend`, `area: security`, `priority: P0`, `demo-critical`

**Work:**

- Emit structured application logs with timestamp, severity, component, event name, and correlation ID.
- Record a minimal audit event for service startup, readiness result, and API error outcome.
- Define redaction for authorization headers, secrets, document text, and user prompts.
- Provide a short troubleshooting guide using the correlation ID.

**Acceptance criteria:**

- [ ] A failed API request can be traced end-to-end with its correlation ID.
- [ ] Logs/audit records do not contain secrets, raw document contents, or raw prompts.
- [ ] Startup and readiness events can be demonstrated from the local log/audit store.

**Dependencies:** FND-04, FND-05.

### FND-07 — Add baseline automated quality checks

**Title:** `FND-07: Add baseline automated quality checks`  
**Labels:** `epic: foundation`, `type: test`, `area: devops`, `priority: P0`

**Work:**

- Add a GitHub Actions workflow (or equivalent documented CI configuration) for formatting, linting, unit tests, and secret scanning.
- Keep CI independent of private model weights and confidential data.
- Fail on test/lint errors and publish concise diagnostics.
- Add status badge/instructions once the repository’s default branch is confirmed.

**Acceptance criteria:**

- [ ] The checks run on pull requests and pushes to `main`.
- [ ] CI passes on the baseline project.
- [ ] A deliberately malformed source file or failing test causes CI to fail.
- [ ] No CI step requires a cloud model API key.

**Dependencies:** FND-01, FND-04.

### FND-08 — Create smoke-test suite and demo readiness check

**Title:** `FND-08: Create smoke-test suite and demo readiness check`  
**Labels:** `epic: foundation`, `type: test`, `area: backend`, `priority: P0`, `demo-critical`

**Work:**

- Test service startup, configuration validation, liveness/readiness, migration bootstrap, and log redaction.
- Add one command that a presenter can run before the demo to validate the base stack.
- Ensure the test suite reports which prerequisite failed, with recovery guidance.

**Acceptance criteria:**

- [ ] The full baseline smoke suite completes on a clean local machine/VM under the documented setup.
- [ ] Each failed dependency produces a specific, actionable diagnosis.
- [ ] The demo readiness command returns a clear pass/fail result.

**Dependencies:** FND-03, FND-04, FND-05, FND-06.

### FND-09 — Write operator and presenter runbooks

**Title:** `FND-09: Write local setup, recovery, and judging-demo runbooks`  
**Labels:** `epic: foundation`, `type: documentation`, `area: platform`, `priority: P1`, `demo-critical`

**Work:**

- Write concise documents for local setup, normal startup, common recovery steps, and controlled shutdown.
- Write a presenter checklist: pre-demo verification, what must be visible, backup narrative if a component fails, and what not to claim.
- Use non-sensitive sample data only.

**Acceptance criteria:**

- [ ] A teammate who did not create the stack can start and verify it from the runbook.
- [ ] The presenter checklist includes the offline/local proof that Epic 2 will supply.
- [ ] The documents distinguish implemented MVP features from future scope.

**Dependencies:** FND-08.

---

# Epic 2 — Sovereign Infrastructure & Offline Proof

**Create as parent issue**

**Title:** `[Epic 2] Sovereign Infrastructure & Offline Proof`

**Milestone:** `M2 — Sovereign Infrastructure & Offline Proof`  
**Labels:** `epic: sovereign-infrastructure`, `priority: P0`, `demo-critical`

## Goal

Deliver local AI infrastructure that can be shown to operate entirely on-premise for the SIH demo: model serving, embeddings, vector storage, hardware awareness, health, and a defendable offline/egress proof.

## MVP boundary

Use a single selected open-weight chat model and one embedding model that fit the available demo hardware. The solution must be configurable and observable; it does not need multi-model orchestration, distributed GPU scheduling, enterprise key management, or production-scale high availability.

## Epic acceptance criteria

- The selected language model and embedding model load from approved local artifact paths.
- All inference/embedding/vector calls remain within the local deployment boundary.
- The system reports model-service status, selected device, basic resource usage, and actionable errors.
- The offline proof can be repeated during the demo without relying on an external API.
- Model/configuration metadata is auditable without storing user content or weights in Git.

## Child issues

### SOV-01 — Select and document the demo model profile

**Title:** `SOV-01: Select and document the local demo model profile`  
**Labels:** `epic: sovereign-infrastructure`, `type: documentation`, `area: ml`, `priority: P0`, `demo-critical`

**Work:**

- Record the chosen chat model, embedding model, quantization/runtime format, minimum RAM/VRAM, and expected capability limits.
- Define a lower-resource fallback profile and the exact trade-off it makes.
- State licensing/redistribution considerations and do not commit weights.
- Set demo-safe parameters such as context length, concurrency, and generation limits.

**Acceptance criteria:**

- [ ] A reviewer can identify the exact local model artifacts required without downloading during runtime.
- [ ] The profile is sized for the actual demo hardware or a documented fallback.
- [ ] Limitations are explicit; unsupported capabilities are not presented as delivered.

**Dependencies:** FND-03.

### SOV-02 — Build local model artifact inventory and import validation

**Title:** `SOV-02: Build local model artifact inventory and import validation`  
**Labels:** `epic: sovereign-infrastructure`, `type: feature`, `area: ml`, `area: devops`, `priority: P0`

**Work:**

- Define an untracked local model artifact directory and manifest format containing model identifier, version, checksum, runtime format, and source/license record.
- Implement startup/preflight validation that verifies required local files before launching a model service.
- Document a controlled offline transfer/import procedure.
- Never store weights, tokens, or download URLs requiring credentials in the repository.

**Acceptance criteria:**

- [ ] A missing or checksum-mismatched artifact stops startup with a clear remediation message.
- [ ] The manifest identifies every artifact used by the demo.
- [ ] A disconnected machine can verify the artifacts without contacting a remote service.

**Dependencies:** SOV-01, FND-02.

### SOV-03 — Deploy local LLM inference service

**Title:** `SOV-03: Deploy local LLM inference service with health checks`  
**Labels:** `epic: sovereign-infrastructure`, `type: feature`, `area: ml`, `area: backend`, `priority: P0`, `demo-critical`

**Work:**

- Run the selected LLM through a local serving runtime behind an internal API endpoint.
- Configure local artifact loading, device selection, bounded request timeouts, and streaming/non-streaming response behavior as required by the planned UI.
- Add health and model-status endpoints without exposing paths, prompts, or raw model configuration to untrusted users.
- Integrate service readiness with the platform API.

**Acceptance criteria:**

- [ ] A locally stored model loads and produces a response through the internal API.
- [ ] Model service health turns unhealthy when the runtime/model is unavailable.
- [ ] The platform readiness result reflects LLM availability.
- [ ] The service functions with the network disconnected after artifacts are present.

**Dependencies:** SOV-01, SOV-02, FND-04.

### SOV-04 — Deploy local embedding service

**Title:** `SOV-04: Deploy local embedding service`  
**Labels:** `epic: sovereign-infrastructure`, `type: feature`, `area: ml`, `priority: P0`

**Work:**

- Serve the selected embedding model locally through an internal interface.
- Enforce a fixed vector dimension/version compatible with the target vector store.
- Add a bounded batch endpoint and clear invalid-input failures.
- Report readiness and model version for later RAG pipeline integration.

**Acceptance criteria:**

- [ ] Text produces a deterministic-dimension embedding locally.
- [ ] The service is unavailable in readiness when the embedding artifact cannot load.
- [ ] No request is sent to a hosted embedding API.
- [ ] The documented dimension matches vector-store configuration.

**Dependencies:** SOV-01, SOV-02, FND-04.

### SOV-05 — Provision persistent local vector store

**Title:** `SOV-05: Provision persistent local vector store`  
**Labels:** `epic: sovereign-infrastructure`, `type: feature`, `area: backend`, `area: devops`, `priority: P0`

**Work:**

- Add one local vector database/service to the Compose stack with persistent storage.
- Define a versioned collection/index naming strategy and metadata schema placeholder for document ID, chunk ID, source location, embedding model version, and timestamp.
- Add a startup connectivity check and a minimal insert/query/delete test using synthetic data.
- Document how local data is cleared for a demo reset without providing a destructive default command.

**Acceptance criteria:**

- [ ] Synthetic vectors can be persisted and retrieved after service restart.
- [ ] The embedding dimension mismatch is detected before writes.
- [ ] The platform readiness check detects vector-store outage.
- [ ] Only local service addresses are accepted for this MVP.

**Dependencies:** FND-02, FND-04, SOV-04.

### SOV-06 — Implement runtime hardware capability detection

**Title:** `SOV-06: Implement CPU/GPU capability detection and safe fallback`  
**Labels:** `epic: sovereign-infrastructure`, `type: feature`, `area: ml`, `area: devops`, `priority: P0`, `demo-critical`

**Work:**

- Detect CPU, available RAM, supported GPU(s), visible VRAM where available, and accelerator/runtime compatibility.
- Select the validated model profile/device policy; allow an explicit safe CPU or smaller-model fallback.
- Surface a clear preflight result when hardware is insufficient rather than attempting a likely crash.
- Keep hardware inventory local; redact machine-specific identifiers from user-facing endpoints.

**Acceptance criteria:**

- [ ] The startup report identifies selected execution mode and whether it meets the chosen profile.
- [ ] A system without a compatible GPU takes the documented fallback path or fails clearly.
- [ ] Hardware information does not expose serial numbers or unrelated host details.

**Dependencies:** SOV-01, FND-03.

### SOV-07 — Add model lifecycle and resource status surface

**Title:** `SOV-07: Add model lifecycle and resource status surface`  
**Labels:** `epic: sovereign-infrastructure`, `type: feature`, `area: backend`, `area: ml`, `priority: P1`

**Work:**

- Track model-service state: not configured, validating, loading, ready, degraded, failed.
- Report coarse CPU/RAM and GPU memory/utilization if the runtime supports it.
- Record state transitions as audit events; expose a presenter-safe status endpoint for the UI.
- Define thresholds that warn rather than silently degrade demo performance.

**Acceptance criteria:**

- [ ] Status changes from loading to ready only after a successful local inference check.
- [ ] A model-load failure shows a human-actionable reason and correlation ID.
- [ ] The presenter-safe endpoint does not disclose model file locations or sensitive host data.

**Dependencies:** SOV-03, SOV-04, SOV-06, FND-06.

### SOV-08 — Enforce local-only endpoints and run offline proof

**Title:** `SOV-08: Enforce local-only service configuration and run offline proof`  
**Labels:** `epic: sovereign-infrastructure`, `type: test`, `area: security`, `area: devops`, `priority: P0`, `demo-critical`

**Work:**

- Maintain an allowlist of local/internal service hostnames/IP ranges for the demo profile.
- Reject cloud inference and embedding endpoint configuration in `offline-demo`.
- Document a repeatable offline test: start with artifacts present, disable/disconnect external networking, run service health and local LLM/embedding/vector smoke calls, capture timestamped evidence.
- Clearly distinguish application-level local-only enforcement from full enterprise firewall/network segmentation, which is future scope unless the host environment supplies it.

**Acceptance criteria:**

- [ ] `offline-demo` refuses a public HTTP(S) AI endpoint configuration.
- [ ] With external connectivity disabled, the local LLM, embedding service, vector store, and platform readiness test pass.
- [ ] The proof produces retained logs/screenshots or terminal output suitable for the SIH demo record.
- [ ] Any attempted non-local configured endpoint is logged as a blocked configuration event without logging secrets.

**Dependencies:** SOV-03, SOV-04, SOV-05, FND-06, FND-08.

### SOV-09 — Package controlled air-gapped bootstrap materials

**Title:** `SOV-09: Package controlled air-gapped bootstrap materials`  
**Labels:** `epic: sovereign-infrastructure`, `type: documentation`, `area: devops`, `priority: P1`

**Work:**

- Produce a checklist of container images, runtime installers, source dependencies, model artifacts, checksums, and licenses needed before disconnection.
- Document import order and verification steps for an isolated machine.
- Include a no-network rehearsal checklist and recovery path for a missing artifact.

**Acceptance criteria:**

- [ ] A reviewer can enumerate what must be transferred before going offline.
- [ ] Every large/binary artifact is referenced by manifest/checksum rather than committed to Git.
- [ ] The sequence does not assume an internet connection after the bootstrap package is prepared.

**Dependencies:** SOV-02, SOV-03, SOV-04, SOV-05, SOV-08.

### SOV-10 — Run and record sovereign infrastructure demo rehearsal

**Title:** `SOV-10: Run and record sovereign infrastructure demo rehearsal`  
**Labels:** `epic: sovereign-infrastructure`, `type: test`, `area: platform`, `priority: P0`, `demo-critical`

**Work:**

- Rehearse the M2 flow on the intended judging hardware: preflight, service startup, status, local inference, local embedding, vector-store persistence, and offline proof.
- Record elapsed startup time, first-response time, failure points, and fallback used.
- Convert any repeated failure into a linked blocking issue before claiming readiness.
- Prepare a concise evidence set that supports the claim: “no runtime external AI API is required.”

**Acceptance criteria:**

- [ ] A complete rehearsal passes on the intended demo setup.
- [ ] The team can explain exactly what ran locally and what evidence verifies it.
- [ ] Known limitations/fallbacks are documented for the presentation.
- [ ] All P0 blockers are either closed or explicitly escalated; no hidden demo risk remains.

**Dependencies:** SOV-03, SOV-04, SOV-05, SOV-06, SOV-08.

---

## Dependency order

```text
FND-01 → FND-02 → FND-03 → FND-04 → FND-05 → FND-06 → FND-08 → FND-09
                         └──────────────→ FND-07

FND-02 + FND-03 → SOV-01 → SOV-02 → SOV-03 ─┐
                                             ├→ SOV-08 → SOV-09
                                  SOV-04 ────┤
                                  SOV-05 ────┤
SOV-01 + FND-03 → SOV-06 ───────────────────┘
SOV-03 + SOV-04 + SOV-06 + FND-06 → SOV-07
SOV-03 + SOV-04 + SOV-05 + SOV-06 + SOV-08 → SOV-10
```

## Recommended board status at creation

Create both parent epics and all child issues in **Backlog**. Move only the P0 critical path into **Todo** in this order:

1. FND-01, FND-02, FND-03
2. FND-04, FND-05, FND-06, FND-07, FND-08
3. SOV-01, SOV-02, SOV-06
4. SOV-03, SOV-04, SOV-05
5. SOV-08, SOV-10

Keep FND-09, SOV-07, and SOV-09 as P1 until the P0 rehearsal succeeds.

## How to create in GitHub

1. Create the two milestones and labels shown above.
2. Create the two parent epic issues, then each child issue, using the titles and bodies in this document.
3. Assign each child to its matching milestone and add its labels.
4. In each parent epic issue, add a task-list link to every child issue created under it.
5. Add issues to a GitHub Project with fields: Status, Priority, Area, Type, and Milestone.

GitHub’s native Issues does not provide a first-class “Epic” type in every repository configuration; the parent issue plus linked child task list is the portable approach.
