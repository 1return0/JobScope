"""Reproducible out-of-domain retrieval check against BEIR SciFact test qrels.

The external corpus is downloaded separately; this script does not alter the
JobScope Current Corpus or claim that SciFact measures recruitment QA quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from app.application.retrieval.dense_retrieval import DenseIndexBuilder
from app.application.retrieval.lexical_retrieval import Bm25Retriever
from app.application.retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion
from app.application.retrieval.retrieval_evaluation import (
    GoldenRetrievalCase,
    RelevanceJudgment,
    RetrievalEvaluator,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation
from app.infrastructure.retrieval.embedding_model_presets import QWEN3_EMBEDDING_06B
from app.infrastructure.retrieval.sentence_transformer_embedder import (
    SentenceTransformerTextEmbedder,
)

EXPECTED_SHA256 = {
    "corpus.jsonl": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
    "queries.jsonl": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
    "test.tsv": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as source:
        return {item["_id"]: item for line in source if (item := json.loads(line))}


def load_dataset(root: Path) -> tuple[list[EvidenceChunk], list[GoldenRetrievalCase], dict[str, str]]:
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels" / "test.tsv"
    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    qrels: dict[str, list[RelevanceJudgment]] = {}
    for line in qrels_path.read_text(encoding="utf-8").splitlines()[1:]:
        query_id, doc_id, grade = line.split("\t")
        if int(grade) > 0:
            qrels.setdefault(query_id, []).append(
                RelevanceJudgment(evidence_id=f"ev_{hashlib.sha256(doc_id.encode()).hexdigest()}", relevance_grade=int(grade))
            )
    chunks: list[EvidenceChunk] = []
    ids: dict[str, str] = {}
    for ordinal, (doc_id, document) in enumerate(sorted(corpus.items())):
        evidence_id = f"ev_{hashlib.sha256(doc_id.encode()).hexdigest()}"
        ids[doc_id] = evidence_id
        title = document.get("title", "").strip()
        body = document.get("text", "").strip()
        text = "\n".join(part for part in (title, body) if part)
        if not text:
            raise ValueError(f"empty BEIR document: {doc_id}")
        chunks.append(EvidenceChunk(
            evidence_id=evidence_id,
            document_sha256=hashlib.sha256(text.encode()).hexdigest(),
            source_reference=f"beir:scifact:{doc_id}",
            source_fragment_ordinal=ordinal,
            chunk_ordinal=0,
            chunker_version="beir-scifact-document-v1",
            text=text,
            location=EvidenceLocation(),
        ))
    cases = [
        GoldenRetrievalCase(
            case_id=query_id,
            query=queries[query_id]["text"],
            judgments=tuple(judgments),
        )
        for query_id, judgments in sorted(qrels.items())
    ]
    if len(chunks) != 5183 or len(cases) != 300:
        raise ValueError(f"unexpected BEIR SciFact size: {len(chunks)} documents, {len(cases)} queries")
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (corpus_path, queries_path, qrels_path)
    }
    if source_hashes != EXPECTED_SHA256:
        raise ValueError("BEIR SciFact source files differ from the frozen snapshot")
    return chunks, cases, source_hashes


def metrics(report):
    return {
        "recall_at_5": round(report.macro_recall_at_k, 6),
        "mrr_at_5": round(report.mean_reciprocal_rank, 6),
        "ndcg_at_5": round(report.macro_ndcg_at_k, 6),
    }


def paired_mrr_interval(left, right, *, resamples: int = 20_000) -> dict[str, float | int]:
    """Paired query bootstrap for the MRR@5 difference (fixed random seed)."""
    differences = np.asarray([
        (1 / a.first_relevant_rank if a.first_relevant_rank else 0.0)
        - (1 / b.first_relevant_rank if b.first_relevant_rank else 0.0)
        for a, b in zip(left.cases, right.cases, strict=True)
    ])
    rng = np.random.default_rng(20261012)
    sampled = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[sampled].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "difference": round(float(differences.mean()), 6),
        "lower": round(float(lower), 6),
        "upper": round(float(upper), 6),
        "resamples": resamples,
        "seed": 20261012,
    }


def _cache_key(source_hashes: dict[str, str], *, precision: str) -> str:
    payload = json.dumps(
        {"source_sha256": source_hashes,
         "embedding_identity": QWEN3_EMBEDDING_06B.identity,
         "precision": precision},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _cached_vectors(path: Path, count: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        vectors = np.load(path, allow_pickle=False)
        if vectors.shape != (count, QWEN3_EMBEDDING_06B.dimension):
            raise ValueError("cached embedding shape mismatch")
        if not np.isfinite(vectors).all():
            raise ValueError("cached embedding contains non-finite values")
        return vectors
    except (OSError, ValueError):
        return None


def _save_vectors(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.save(output, vectors, allow_pickle=False)
    temporary.replace(path)


def embed_corpus(
    chunks: list[EvidenceChunk], embedder: SentenceTransformerTextEmbedder,
    cache_dir: Path, *, block_size: int,
) -> tuple[list[str], np.ndarray]:
    ordered = sorted(chunks, key=lambda chunk: chunk.evidence_id)
    vectors: list[np.ndarray] = []
    for start in range(0, len(ordered), block_size):
        block = ordered[start:start + block_size]
        cache_path = cache_dir / f"documents-{start:05d}.npy"
        cached = _cached_vectors(cache_path, len(block))
        if cached is None:
            index = DenseIndexBuilder(embedder).build(block)
            cached = np.asarray([record.vector for record in index.records], dtype=np.float32)
            _save_vectors(cache_path, cached)
        vectors.append(cached)
        print(f"Documents: {start + len(block)}/{len(ordered)}", flush=True)
    return [chunk.evidence_id for chunk in ordered], np.vstack(vectors)


def embed_queries(
    cases: list[GoldenRetrievalCase], embedder: SentenceTransformerTextEmbedder,
    cache_dir: Path, *, block_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(cases), block_size):
        block = cases[start:start + block_size]
        cache_path = cache_dir / f"queries-{start:05d}.npy"
        cached = _cached_vectors(cache_path, len(block))
        if cached is None:
            cached = np.asarray([embedder.embed_query(case.query) for case in block], dtype=np.float32)
            _save_vectors(cache_path, cached)
        vectors.append(cached)
        print(f"Queries: {start + len(block)}/{len(cases)}", flush=True)
    result = np.vstack(vectors)
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/external/scifact"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/beir-scifact-test.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("evaluation/external/embedding-cache"))
    args = parser.parse_args()
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("fp16 precision requires CUDA")
    if args.block_size < 1:
        parser.error("block-size must be positive")
    chunks, cases, source_hashes = load_dataset(args.dataset)
    print(f"Loaded {len(chunks)} documents and {len(cases)} test claims", flush=True)
    embedder = SentenceTransformerTextEmbedder(
        QWEN3_EMBEDDING_06B, device=args.device,
        batch_size=args.batch_size, cache_folder=Path(".model-cache"),
        local_files_only=True,
    )
    if args.precision == "fp16":
        embedder._model.half()  # Inference-only; precision is recorded in the report.
    cache_dir = args.cache_dir / _cache_key(source_hashes, precision=args.precision)
    bm25 = Bm25Retriever(chunks)
    print("Embedding document corpus", flush=True)
    ordered_ids, matrix = embed_corpus(chunks, embedder, cache_dir, block_size=args.block_size)
    print("Embedding test claims", flush=True)
    query_vectors = embed_queries(cases, embedder, cache_dir, block_size=args.block_size)
    scores = query_vectors @ matrix.T
    print("Ranking and evaluating", flush=True)
    bm25_rankings = [
        [hit.chunk.evidence_id for hit in bm25.search(case.query, top_k=20)]
        for case in cases
    ]
    dense_rankings = [
        [ordered_ids[i] for i in np.argsort(-row, kind="stable")[:20]]
        for row in scores
    ]
    fused_rankings = [
        [hit.evidence_id for hit in reciprocal_rank_fusion(
            {"bm25": lexical, "dense": semantic}, top_k=5, rank_constant=60,
        )]
        for lexical, semantic in zip(bm25_rankings, dense_rankings, strict=True)
    ]
    evaluator = RetrievalEvaluator()
    def evaluate(rankings):
        by_query = {case.query: ranking for case, ranking in zip(cases, rankings, strict=True)}
        return evaluator.evaluate(cases, retrieve=lambda query, top_k: by_query[query][:top_k], top_k=5)
    baseline = evaluate(bm25_rankings)
    dense = evaluate(dense_rankings)
    hybrid = evaluate(fused_rankings)
    payload = {
        "dataset": "BEIR SciFact test", "source": "https://github.com/beir-cellar/beir/wiki/Datasets-available",
        "qrels": "published BEIR test qrels; not JobScope-generated labels",
        "domain_boundary": "English scientific claim retrieval, not Chinese recruitment QA or end-to-end Agent quality",
        "document_unit": "one BEIR corpus record (title plus abstract); no JobScope parsing/chunking",
        "documents": len(chunks), "queries": len(cases), "top_k": 5,
        "source_sha256": source_hashes,
        "embedding_model": QWEN3_EMBEDDING_06B.model_name,
        "embedding_revision": QWEN3_EMBEDDING_06B.model_revision,
        "embedding_identity": QWEN3_EMBEDDING_06B.identity,
        "inference_precision": args.precision,
        "embedding_cache_key": cache_dir.name,
        "query_instruction": QWEN3_EMBEDDING_06B.query_instruction,
        "bm25": metrics(baseline), "dense": metrics(dense), "hybrid_rrf": metrics(hybrid),
        "hybrid_minus_bm25": {key: round(metrics(hybrid)[key] - metrics(baseline)[key], 6) for key in metrics(baseline)},
        "paired_mrr_at_5_bootstrap_95ci": {
            "hybrid_minus_bm25": paired_mrr_interval(hybrid, baseline),
            "hybrid_minus_dense": paired_mrr_interval(hybrid, dense),
        },
        "per_query": [
            {"query_id": case.case_id, "bm25_first_rank": b.first_relevant_rank,
             "dense_first_rank": d.first_relevant_rank, "hybrid_first_rank": h.first_relevant_rank}
            for case, b, d, h in zip(cases, baseline.cases, dense.cases, hybrid.cases, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("documents", "queries", "bm25", "dense", "hybrid_rrf", "hybrid_minus_bm25")}, indent=2))


if __name__ == "__main__":
    main()
