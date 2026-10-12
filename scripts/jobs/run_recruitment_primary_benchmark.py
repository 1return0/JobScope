"""Recruitment-domain retrieval, SQL filtering and real Job Agent evaluation.

All inputs are frozen before scoring. Production Corpus and Checkpoint tables
are untouched: PostgreSQL work takes place in a newly created private schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, date, datetime
from html import escape
from pathlib import Path

import httpx
from pydantic import ValidationError
from sqlalchemy import text

from app.application.jobs.job_search_tool import SearchCurrentJobsArguments, SearchCurrentJobsTool
from app.application.retrieval.dense_retrieval import DenseIndexBuilder, InMemoryDenseRetriever
from app.application.retrieval.lexical_retrieval import Bm25Retriever
from app.application.retrieval.retrieval_evaluation import GoldenRetrievalCase, RelevanceJudgment, RetrievalEvaluator
from app.domain.documents.document_chunking import ChunkingConfig, StructureAwareChunker
from app.domain.documents.document_ingestion import DocumentArtifact
from app.infrastructure.documents.html_document_parser import HtmlDocumentParser
from app.infrastructure.persistence.database import Base, create_session_factory
from app.infrastructure.persistence.document_corpus_repository import SqlAlchemyDocumentCorpusRepository
from app.infrastructure.persistence.models import (
    DocumentSnapshotChunkRow, DocumentSnapshotRow, EvidenceChunkRow, StructuredJobRecordRow,
)
from app.infrastructure.persistence.structured_job_record_query import SqlAlchemyCurrentJobRecordQuery

ROOT = Path("evaluation/datasets/recruitment-primary-v1")
EXTERNAL = Path("evaluation/external/recruitment-primary-v1")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_fixture():
    raw = (ROOT / "benchmark.json").read_bytes()
    fixture = json.loads(raw)
    if fixture["human_reviewed"] or fixture["annotation_status"] != "source_grounded_agent_authored":
        raise ValueError("annotation provenance is incorrect")
    if fixture["annotations_sha256"] != sha((ROOT / "annotations.json").read_bytes()):
        raise ValueError("annotations changed after source verification")
    for key in ("retrieval_cases", "structured_cases", "agent_cases"):
        ids = [row["case_id"] for row in fixture[key]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("case IDs must be present and unique")
    for case in fixture["structured_cases"]:
        SearchCurrentJobsArguments.model_validate(case["filters"])
    return fixture, sha(raw)


def build_corpus(fixture, output: Path):
    """Use the existing HTML parser and chunker on normalized public summaries."""
    groups = defaultdict(list)
    for document in fixture["documents"]:
        groups[document["source_id"]].append(document)
    chunks, units, pages, document_pages = [], {}, {}, {}
    output.mkdir(parents=True, exist_ok=True)
    labels = {"metadata": "岗位信息", "duties": "工作职责", "requirements": "任职要求"}
    for source_id, documents in groups.items():
        url = documents[0]["source_url"]
        title = documents[0]["title"] if len(documents) == 1 else source_id + "公开招聘资料快照"
        body = [f"<h1>{escape(title)}</h1>"]
        text_to_unit = {}
        for document in documents:
            for section, content in document["sections"].items():
                label = labels.get(section, document["title"])
                body += [f"<h2>{escape(label)}</h2>", f"<p>{escape(content)}</p>"]
                normalized = " ".join(content.split())
                if normalized in text_to_unit:
                    raise ValueError("duplicate normalized evidence text")
                text_to_unit[normalized] = document["id"] + ":" + section
            document_pages[document["id"]] = source_id
        raw = ("<!doctype html><html><head><meta charset='utf-8'></head><body>"
               + "\n".join(body) + "</body></html>").encode("utf-8")
        path = output / (source_id + ".html")
        path.write_bytes(raw)
        artifact = DocumentArtifact(path, url, "html", sha(raw), len(raw))
        parsed = HtmlDocumentParser().parse(artifact)
        page_chunks = StructureAwareChunker().chunk(parsed)
        for chunk in page_chunks:
            unit = text_to_unit.get(chunk.text)
            if unit:
                units[unit] = chunk.evidence_id
        chunks.extend(page_chunks)
        pages[source_id] = {"artifact": artifact, "chunks": page_chunks,
                            "snapshot_id": "snap_" + sha((url + artifact.content_sha256).encode())}
    for case in fixture["retrieval_cases"]:
        if any(unit not in units for unit in case["target_units"]):
            raise ValueError("golden evidence unit is absent after real parsing/chunking")
    return chunks, units, pages, document_pages


def populate_fixture(engine, fixture, chunks, pages, document_pages):
    """Load source-authored fixture facts, not model extraction outputs."""
    Base.metadata.create_all(engine)
    now = datetime(2026, 10, 12, tzinfo=UTC)
    session_factory = create_session_factory(engine)
    record_aliases = {}
    with session_factory.begin() as session:
        for page in pages.values():
            artifact = page["artifact"]
            session.add(DocumentSnapshotRow(
                snapshot_id=page["snapshot_id"], source_reference=artifact.source_reference,
                document_format="html", content_sha256=artifact.content_sha256, byte_size=artifact.byte_size,
                version_number=1, first_observed_at=now, last_observed_at=now, is_current=True))
        session.flush()
        for chunk in chunks:
            location = chunk.location
            session.add(EvidenceChunkRow(
                evidence_id=chunk.evidence_id, document_sha256=chunk.document_sha256,
                source_reference=chunk.source_reference, source_fragment_ordinal=chunk.source_fragment_ordinal,
                chunk_ordinal=chunk.chunk_ordinal, chunker_version=chunk.chunker_version,
                text=chunk.text, heading_path=list(location.heading_path), page_number=location.page_number,
                sheet_name=location.sheet_name, cell_reference=location.cell_reference,
                slide_number=location.slide_number, table_number=location.table_number,
                table_row_number=location.table_row_number))
        session.flush()
        for page in pages.values():
            for chunk in page["chunks"]:
                session.add(DocumentSnapshotChunkRow(snapshot_id=page["snapshot_id"], evidence_id=chunk.evidence_id))
        for row in fixture["job_records"]:
            record_id = "job_" + sha(row["id"].encode())
            record_aliases[record_id] = row["id"]
            deadline = date.fromisoformat(row["application_deadline"]) if row["application_deadline"] else None
            session.add(StructuredJobRecordRow(
                record_id=record_id, record_fingerprint=sha(json.dumps(row, sort_keys=True).encode()),
                source_snapshot_id=pages[document_pages[row["document_id"]]]["snapshot_id"],
                extraction_contract_version="recruitment-benchmark-fixture-v1",
                prompt_version="not-model-extracted", model_identity="source-grounded fixture",
                judge_identity="not independently human reviewed", record_payload=row,
                company=row["company"], job_title=row["job_title"], locations=row["locations"],
                education_requirement=row["education_requirement"], major_requirement=None,
                recruitment_type=row["recruitment_type"], application_deadline_raw=row["application_deadline"],
                application_deadline_normalized=deadline,
                deadline_normalization_status="normalized" if deadline else "not_stated",
                deadline_reason_code=None, date_normalizer_version="job-date-v1",
                is_current=True, activated_at=now, activation_reference="isolated benchmark fixture"))
    return session_factory, record_aliases


def evaluate_sql(fixture, session_factory, aliases):
    tool = SearchCurrentJobsTool(SqlAlchemyCurrentJobRecordQuery(session_factory))
    results = []
    for case in fixture["structured_cases"]:
        output = tool.invoke(case["filters"])
        actual = sorted(aliases[job.record_id] for job in output.jobs)
        expected = sorted(case["expected_ids"])
        results.append({"case_id": case["case_id"], "expected_ids": expected, "actual_ids": actual,
                        "status": output.status, "passed": actual == expected})
    unsupported = []
    for field in fixture["unsupported_argument_cases"]:
        try:
            tool.invoke({field: "unsupported benchmark value"})
            rejected = False
        except ValidationError:
            rejected = True
        unsupported.append({"field": field, "rejected": rejected})
    return {"cases": len(results), "passed": sum(row["passed"] for row in results), "results": results,
            "unsupported_argument_results": unsupported,
            "boundary": "real query/tool on source-grounded fixture records; does not measure model extraction accuracy"}, tool


@contextmanager
def isolated_postgres(settings):
    from app.infrastructure.persistence.database import create_postgresql_engine
    base = create_postgresql_engine(settings)
    schema = "jobscope_benchmark_" + uuid.uuid4().hex
    if re.fullmatch(r"jobscope_benchmark_[0-9a-f]{32}", schema) is None:
        raise ValueError("invalid isolated schema name")
    created = False
    try:
        with base.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        yield base.execution_options(schema_translate_map={None: schema})
    finally:
        if created:
            # Only the exact random schema created by this invocation is dropped.
            with base.begin() as connection:
                connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        base.dispose()


class CachedEmbedder:
    def __init__(self, delegate, cache):
        self.delegate, self.cache = delegate, cache

    @property
    def spec(self):
        return self.delegate.spec

    def _vectors(self, kind, texts, build):
        key = sha(json.dumps({"kind": kind, "texts": texts, "identity": self.spec.identity}, ensure_ascii=False).encode())
        path = self.cache / (key + ".json")
        if path.exists():
            values = json.loads(path.read_text(encoding="utf-8"))
        else:
            values = build()
            atomic_json(path, values)
        if len(values) != len(texts) or any(len(v) != self.spec.dimension or not all(math.isfinite(x) for x in v) for v in values):
            raise ValueError("cached embedding shape or values are invalid")
        return tuple(tuple(v) for v in values)

    def embed_documents(self, texts):
        return self._vectors("documents", texts, lambda: self.delegate.embed_documents(texts))

    def embed_query(self, query):
        return self._vectors("query", (query,), lambda: (self.delegate.embed_query(query),))[0]


def cluster_mrr_interval(left, right, cases):
    groups = defaultdict(list)
    differences = {a.case_id: a.reciprocal_rank - b.reciprocal_rank
                   for a, b in zip(left.cases, right.cases, strict=True)}
    for case in cases:
        groups[case["source_group"]].append(differences[case["case_id"]])
    keys = sorted(groups)
    sums = {key: sum(groups[key]) for key in keys}
    rng, samples = random.Random(20261012), []
    for _ in range(10_000):
        picked = rng.choices(keys, k=len(keys))
        samples.append(sum(sums[key] for key in picked) / sum(len(groups[key]) for key in picked))
    samples.sort()
    return {"difference": sum(differences.values()) / len(differences), "lower": samples[249],
            "upper": samples[9749], "source_groups": len(keys), "seed": 20261012, "resamples": 10_000}


def evaluate_retrieval(fixture, chunks, units, searcher, dense, lexical, search_tool):
    cases = [GoldenRetrievalCase(row["case_id"], row["query"],
             tuple(RelevanceJudgment(units[unit]) for unit in row["target_units"])) for row in fixture["retrieval_cases"]]
    evaluator, reports = RetrievalEvaluator(), {}
    rankings = {}
    for name in ("bm25", "dense", "hybrid"):
        print(f"Scoring recruitment retrieval: {name}", flush=True)
        by_query = {}
        for number, case in enumerate(cases, start=1):
            if name == "bm25":
                ids = [hit.chunk.evidence_id for hit in lexical.search(case.query, top_k=5)]
            elif name == "dense":
                ids = [hit.chunk.evidence_id for hit in dense.search(case.query, top_k=5)]
            else:
                ids = [hit.chunk.evidence_id for hit in searcher.search(case.query, top_k=5).results]
            by_query[case.query] = ids
            if number % 20 == 0:
                print(f"{name}: {number}/{len(cases)}", flush=True)
        reports[name] = evaluator.evaluate(cases, retrieve=lambda query, k: by_query[query][:k], top_k=5)
        rankings[name] = by_query
    scoped_rows = [row for row in fixture["retrieval_cases"] if "scope_filters" in row]
    scoped_cases = [case for case in cases if case.case_id in {row["case_id"] for row in scoped_rows}]
    scoped_rankings, source_checks = {}, []
    for row in scoped_rows:
        output = search_tool.invoke(row["scope_filters"])
        scope = tuple(sorted({job.source_reference for job in output.jobs if job.source_reference}))
        hits = searcher.search(row["query"], top_k=5, source_references=scope).results
        scoped_rankings[row["query"]] = [hit.chunk.evidence_id for hit in hits]
        source_checks.append({"case_id": row["case_id"], "source_scope": scope,
                              "source_scope_valid": all(hit.chunk.source_reference in scope for hit in hits)})
    scoped = evaluator.evaluate(scoped_cases, retrieve=lambda query, k: scoped_rankings[query][:k], top_k=5)
    unscoped = evaluator.evaluate(scoped_cases, retrieve=lambda query, k: rankings["hybrid"][query][:k], top_k=5)
    summary = {name: asdict(report) for name, report in reports.items()}
    for name, report in reports.items():
        summary[name]["strata"] = {
            kind: {metric: sum(getattr(case, metric) for case in report.cases if meta[case.case_id] == kind)
                   / sum(value == kind for value in meta.values())
                   for metric in ("recall_at_k", "reciprocal_rank", "ndcg_at_k")}
            for meta in [{row["case_id"]: row["stratum"] for row in fixture["retrieval_cases"]}]
            for kind in sorted({row["stratum"] for row in fixture["retrieval_cases"]})
        }
    return {"global": summary, "scoped_subset": {"unscoped_hybrid": asdict(unscoped), "sql_scoped_hybrid": asdict(scoped),
             "source_checks": source_checks, "mrr_difference_95ci": cluster_mrr_interval(scoped, unscoped, scoped_rows)},
            "global_mrr_difference_95ci": {
                "dense_minus_bm25": cluster_mrr_interval(reports["dense"], reports["bm25"], fixture["retrieval_cases"]),
                "hybrid_minus_bm25": cluster_mrr_interval(reports["hybrid"], reports["bm25"], fixture["retrieval_cases"])}}


class RecordingCompletion:
    def __init__(self, delegate, cache, model):
        self.delegate, self.cache, self.model = delegate, cache, model
        self.calls = 0

    def __call__(self, **kwargs):
        from types import SimpleNamespace
        key = sha(json.dumps(kwargs, ensure_ascii=False, sort_keys=True).encode())
        path = self.cache / (key + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            usage = SimpleNamespace(**saved["usage"]) if saved["usage"] else None
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=saved["content"]))], usage=usage)
        self.calls += 1
        response = self.delegate(**kwargs)
        usage = response.usage.model_dump() if response.usage else None
        atomic_json(path, {"content": response.choices[0].message.content, "usage": usage})
        return response


def constraints_retained(actual, expected):
    for key, value in expected.items():
        candidate = actual.get(key, 10 if key == "limit" else None)
        if key == "recruitment_type" and candidate is not None:
            from app.domain.jobs.recruitment_type import normalize_recruitment_type
            try:
                candidate = normalize_recruitment_type(candidate)
                value = normalize_recruitment_type(value)
            except ValueError:
                return False
        if key == "location" and candidate is not None:
            candidate = "".join(candidate.split()).removesuffix("市")
            value = "".join(value.split()).removesuffix("市")
        if key == "job_title" and isinstance(candidate, str):
            candidate, value = candidate.casefold(), value.casefold()
            if re.fullmatch(r"j\d+", value) and value in candidate:
                continue
        if candidate != value:
            return False
    return True


def evaluate_agent(fixture, searcher, search_tool, aliases, units, settings, cache):
    from openai import OpenAI
    from app.application.answering.current_corpus_answering import CurrentCorpusAnswerService
    from app.application.answering.grounded_answer_service import GroundedAnswerService
    from app.application.jobs.job_agent_graph import build_job_agent_graph
    from app.application.jobs.job_rag_tools import AnswerRecruitmentQuestionTool, SearchJobsWithEvidenceTool
    from app.infrastructure.llm.openai_compatible_answer_model import OpenAiCompatibleAnswerModel, OpenAiCompatibleAnswerModelConfig
    from app.infrastructure.llm.openai_compatible_job_agent_planner import OpenAiCompatibleJobAgentPlanner, OpenAiCompatibleJobAgentPlannerConfig
    client = OpenAI(api_key=os.environ[settings.answer_model_api_key_environment_variable],
                    base_url=settings.answer_model_base_url, timeout=settings.answer_model_timeout_seconds,
                    max_retries=0, http_client=httpx.Client(trust_env=False))
    completion = RecordingCompletion(client.chat.completions.create, cache / "completions", settings.answer_model_name)
    planner = OpenAiCompatibleJobAgentPlanner(completion, OpenAiCompatibleJobAgentPlannerConfig(
        model_name=settings.answer_model_name, base_url=settings.answer_model_base_url,
        api_key_environment_variable=settings.answer_model_api_key_environment_variable,
        timeout_seconds=settings.answer_model_timeout_seconds, max_completion_tokens=800))
    answer_model = OpenAiCompatibleAnswerModel(completion, OpenAiCompatibleAnswerModelConfig(
        model_name=settings.answer_model_name, base_url=settings.answer_model_base_url,
        api_key_environment_variable=settings.answer_model_api_key_environment_variable,
        timeout_seconds=settings.answer_model_timeout_seconds, max_completion_tokens=1200))
    answer_tool = AnswerRecruitmentQuestionTool(CurrentCorpusAnswerService(searcher, GroundedAnswerService(answer_model)))
    tools = (search_tool, answer_tool, SearchJobsWithEvidenceTool(search_tool, answer_tool))
    graph = build_job_agent_graph(planner, tools)
    results = []
    try:
        for number, case in enumerate(fixture["agent_cases"], start=1):
            path = cache / "agent-results" / (case["case_id"] + ".json")
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                try:
                    state = graph.invoke({"user_query": case["query"], "owner_id": "recruitment-benchmark"})
                    call = state.get("tool_call") or {}
                    output = (state.get("tool_result") or {}).get("output", {})
                    row = {"case_id": case["case_id"], "mode": case["mode"], "query": case["query"],
                           "planning_status": state.get("planning_status"), "tool_name": call.get("name"),
                           "actual_arguments": call.get("arguments", {}), "tool_error": state.get("tool_error"),
                           "output": output, "plan_reason": state.get("plan_reason")}
                    if case["mode"] == "refuse":
                        row["passed"] = state.get("planning_status") == "refused"
                    else:
                        route_ok = call.get("name") == case["expected_tool"]
                        retained = constraints_retained(call.get("arguments", {}), case.get("expected_filters", {}))
                        row.update({"routing_correct": route_ok, "constraints_retained": retained})
                        checks = [route_ok, retained, state.get("tool_error") is None]
                        if case["mode"] in ("search", "mixed"):
                            jobs = output.get("jobs", []) if case["mode"] == "search" else output.get("jobs", {}).get("jobs", [])
                            actual_ids = sorted(aliases[job["record_id"]] for job in jobs)
                            row["actual_ids"] = actual_ids
                            row["job_set_correct"] = actual_ids == sorted(case["expected_ids"])
                            checks.append(row["job_set_correct"])
                        if case["mode"] in ("answer", "mixed"):
                            answer = output if case["mode"] == "answer" else (output.get("evidence_answer") or {})
                            citations = [citation for claim in answer.get("claims", []) for citation in claim.get("citations", [])]
                            expected_units = {units[unit] for unit in case["target_units"]}
                            quoted = " ".join(citation["quoted_text"] for citation in citations)
                            quoted = quoted.replace("二十四", "24").replace("下午三点", "15")
                            row["answer_status_correct"] = answer.get("status") == case["expected_answer"]
                            row["citation_grounding_valid"] = answer.get("grounding_valid") is True
                            row["target_evidence_cited"] = expected_units.issubset({citation["evidence_id"] for citation in citations})
                            row["required_fact_evidence_present"] = all(term in quoted for term in case["required_terms"])
                            checks.extend([row["answer_status_correct"], row["citation_grounding_valid"],
                                           row["target_evidence_cited"], row["required_fact_evidence_present"]])
                            if case["mode"] == "mixed":
                                scope = output.get("source_scope", [])
                                row["citation_scope_valid"] = all(citation["source_reference"] in scope for citation in citations)
                                checks.append(row["citation_scope_valid"])
                        row["passed"] = all(checks)
                except Exception as error:
                    row = {"case_id": case["case_id"], "mode": case["mode"], "passed": False,
                           "execution_error_type": type(error).__name__}
                atomic_json(path, row)
            results.append(row)
            print(f"Agent {number}/{len(fixture['agent_cases'])}: {case['case_id']} passed={row['passed']}", flush=True)
    finally:
        client.close()
    return {"cases": len(results), "passed": sum(row["passed"] for row in results),
            "model_identity": planner.identity, "fresh_http_calls": completion.calls,
            "results": results,
            "boundary": "fresh stateless Graph, real planner/answer models and real SQL; citation checks are structural/slot checks, not independent semantic answer judging"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/recruitment-primary-v1-real.json"))
    args = parser.parse_args()
    fixture, fixture_hash = load_fixture()
    cache = EXTERNAL / "runs" / fixture_hash
    chunks, units, pages, document_pages = build_corpus(fixture, cache / "normalized-html")
    if args.validate_only:
        print(json.dumps({"documents": len(pages), "chunks": len(chunks), "retrieval_cases": len(fixture["retrieval_cases"]),
                          "sql_cases": len(fixture["structured_cases"]), "agent_cases": len(fixture["agent_cases"]),
                          "dataset_sha256": fixture_hash}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError("completed report must not be overwritten")
    from app.config import load_settings
    from app.application.retrieval.current_corpus_hybrid_search import CurrentCorpusHybridSearchService
    from app.infrastructure.retrieval.embedding_model_presets import QWEN3_EMBEDDING_06B
    from app.infrastructure.retrieval.sentence_transformer_embedder import SentenceTransformerTextEmbedder
    settings = load_settings()
    config = {"embedding_identity": QWEN3_EMBEDDING_06B.identity, "precision": "fp32",
              "planner_model": settings.answer_model_name, "planner_endpoint_sha256": sha(settings.answer_model_base_url.encode()),
              "candidate_k": settings.hybrid_candidate_k, "rank_constant": settings.hybrid_rank_constant,
              "code_sha256": {name: sha(Path(name).read_bytes()) for name in (
                  "app/application/retrieval/lexical_retrieval.py", "app/application/retrieval/current_corpus_hybrid_search.py",
                  "app/infrastructure/persistence/structured_job_record_query.py", "app/application/jobs/job_search_tool.py",
                  "app/application/jobs/job_agent_graph.py", "app/application/jobs/job_agent_planner_prompting.py",
                  "scripts/jobs/run_recruitment_primary_benchmark.py")}}
    model_cache = cache / sha(json.dumps(config, sort_keys=True).encode())
    report = {"dataset_id": fixture["dataset_id"], "dataset_sha256": fixture_hash,
              "annotation_status": fixture["annotation_status"], "human_reviewed": False,
              "as_of_date": fixture["as_of_date"], "evaluated_at": datetime.now(UTC).isoformat(),
              "original_source_pages": len(fixture["sources"]), "normalized_html_documents": len(pages),
              "chunks": len(chunks), "job_records": len(fixture["job_records"]), "config": config}
    with isolated_postgres(settings) as engine:
        session_factory, aliases = populate_fixture(engine, fixture, chunks, pages, document_pages)
        report["structured_filtering"], search_tool = evaluate_sql(fixture, session_factory, aliases)
        report["database_backend"] = "postgresql (isolated temporary schema, removed after run)"
        print(f"Real SQL: {report['structured_filtering']['passed']}/{report['structured_filtering']['cases']}", flush=True)
        embedder = CachedEmbedder(SentenceTransformerTextEmbedder(
            QWEN3_EMBEDDING_06B, device="cuda", batch_size=4,
            cache_folder=Path(".model-cache"), local_files_only=True), model_cache / "embeddings")
        reader = SqlAlchemyDocumentCorpusRepository(session_factory)
        searcher = CurrentCorpusHybridSearchService(reader, embedder, chunker_version=ChunkingConfig().identity,
                    candidate_k=settings.hybrid_candidate_k, rank_constant=settings.hybrid_rank_constant)
        searcher.prepare()
        dense = InMemoryDenseRetriever(DenseIndexBuilder(embedder).build(chunks), embedder)
        report["corpus_fingerprint"] = dense.corpus_fingerprint
        report["retrieval"] = evaluate_retrieval(fixture, chunks, units, searcher, dense, Bm25Retriever(chunks), search_tool)
        atomic_json(model_cache / "retrieval-complete.json", report)
        report["agent"] = evaluate_agent(fixture, searcher, search_tool, aliases, units, settings, model_cache)
    report["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(args.output), "sql_passed": report["structured_filtering"]["passed"],
                      "agent_passed": report["agent"]["passed"], "agent_cases": report["agent"]["cases"],
                      "retrieval": {name: {key: value for key, value in values.items() if key not in ("cases", "strata")}
                                    for name, values in report["retrieval"]["global"].items()}}, indent=2))


if __name__ == "__main__":
    main()
