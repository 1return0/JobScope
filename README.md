# JobScope

JobScope turns public recruitment materials into versioned, evidence-backed job facts and searchable answers. It combines multi-format document ingestion, structured job extraction, hybrid retrieval, and a LangGraph job-search workflow.

## What it does

- Imports and parses HTML, PDF, DOCX, XLSX, and PPTX recruitment materials while retaining source references, hashes, page or cell locations, and document snapshots.
- Uses OCR as a PDF fallback for scanned pages, with a small golden dataset and CER/WER and critical-field evaluation tools.
- Extracts structured job facts and validates their evidence against the source chunks before storing them for SQL-backed search.
- Combines BM25 and dense retrieval with reciprocal-rank fusion; SQL-selected source scopes keep matched-job evidence together, and grounded answers retain citations or return an insufficient-evidence result.
- Provides a LangGraph job-search workflow that calls structured search tools, plus a Vue interface for document preview, search, Agent queries, source review, and evaluation artifacts.

## Architecture

```text
Recruitment files
  → Artifact / Parser / ParsedDocument
  → Article / Fragment / Chunk
  → versioned corpus and evidence-backed job facts
  → SQL filters + source scope + BM25/Dense + RRF
  → cited answer or an explicit insufficient-evidence result
```

FastAPI application code is under `app/`; `frontend/` contains the Vue 3 client. The browser upload screen is an intentionally temporary parse/chunk preview and does not write to the Current Corpus. Persistent corpus preparation uses the verified manifest-based ingestion workflow under `scripts/corpus/`. Architecture and module navigation are documented in [`docs/project-structure.md`](docs/project-structure.md).

The final capability boundaries, demo sequence, and verification commands are in
[`docs/final-delivery.md`](docs/final-delivery.md).

## Run locally

Use Python 3.12. Copy `.env.example` to `.env`, then configure a local PostgreSQL database and any model credentials you want to use. Do not commit `.env`.

```powershell
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8110
```

In another terminal, start the Vue client:

```powershell
cd frontend
npm install
npm run dev
```

Optional dependency groups are available for local embedding models, answer generation, and PaddleOCR. See `pyproject.toml` before installing those larger packages.

Run the automated tests with:

```powershell
py -3.12 -m unittest discover -s tests -v
```

Corpus manifest tools are in `scripts/corpus/`. They validate source metadata and local artifact hashes before ingestion.

## Verification boundaries

The primary recruitment-domain evaluation freezes 40 public official job postings,
recruiting FAQs and a dated recruitment notice: 43 source pages, 337 parsed chunks,
134 retrieval questions, 32 SQL filtering cases and 32 real Agent tasks.

On the same 40 explicit-posting questions, SQL-selected source scoping raised
Recall@5 from **45.0% to 97.5%** and MRR@5 from **0.322 to 0.565**, compared with
unscoped Hybrid retrieval. Real PostgreSQL filtering passed **32/32** cases;
the real Planner/Graph/tool task evaluation passed **26/32** under the frozen
strict contract. The full 134-question BM25/Dense/RRF comparisons, source-group
confidence intervals and failure cases are retained in the
[primary evaluation report](docs/benchmarks/recruitment-primary-v1.md).

The dataset is source-grounded and Agent-authored, without independent human
annotation review; the corpus consists of derived summaries, not an exhaustive
job-market crawl. SQL tests use fixed source-backed fixture facts and do not
measure model extraction accuracy. These offline results do not establish
production accuracy, throughput or reliability. The
[BEIR SciFact report](docs/benchmarks/beir-scifact.md) remains a supplementary
English scientific retrieval experiment, not a recruitment result.
