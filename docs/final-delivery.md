# JobScope final delivery

## Scope

JobScope is an evidence-backed recruitment intelligence application. It imports
recruitment material, retains verifiable evidence, supports SQL filtering plus
hybrid retrieval, and routes natural-language requests through a bounded
LangGraph Job Agent.

```text
Source file → Artifact → ParsedDocument → Article → Fragment / Chunk
           → current facts (SQL) + BM25 / Dense / RRF / optional rerank
           → cited answer or insufficient evidence

Browser session → trusted owner → derived checkpoint thread
                → Job Agent Planner → registered read-only tool → safe response
```

## Demo sequence

1. Open **Documents** and upload a non-sensitive recruitment file for preview.
2. Open **Sources** to inspect source references and review status.
3. Open **RAG** and ask a question that has evidence; inspect citations.
4. Ask a question without evidence; show `insufficient_evidence` rather than
   invented content.
5. Open **Job Agent**, run a read-only search, inspect the selected tool and
   execution Harness. Refresh to show the bounded safe transcript.
6. Save a preference explicitly, then remove it. Clear the conversation and
   show that preferences are a separate store.

## Verification commands

```powershell
# Backend focused regression
.\.venv\Scripts\python.exe -m pytest tests\jobs tests\api tests\answering

# Vue production build
cd frontend
npm run build
```

Real PostgreSQL checks are opt-in and require a configured database. They are
not implied by the normal test command:

```powershell
$env:JOBSCOPE_RUN_POSTGRES_JOB_AGENT_MEMORY_SMOKE = "true"
.\.venv\Scripts\python.exe -m pytest tests\jobs\test_job_agent_checkpoint_runtime_smoke.py
```

## Recorded VM verification (2026-10-02)

With the configured PostgreSQL VM available, the opt-in checks for the
PostgreSQL Checkpointer, consented preference store, browser-session ownership,
and Job Agent checkpoint recovery passed: **5 passed**. The Smoke uses isolated
owners/threads and removes its test records. This verifies persistence and
ownership boundaries in that environment; it is not a load, availability, or
live-model quality claim.

## Claims and boundaries

The project demonstrates bounded Agent tool routing, source-scoped retrieval,
safe checkpoint conversation history, consented preferences, controlled
read-only retrieval expansion, model/tool/time budgets, and stable failure
mapping.

It does **not** claim production accuracy, multi-tenant billing, global rate
limits across multiple processes, unrestricted autonomous actions, or verified
performance under production traffic. A configured real-model run is required
before claiming live model quality or cost characteristics.
