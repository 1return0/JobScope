from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.application.retrieval.dense_retrieval_experiment import (
    DenseRetrievalExperimentReport,
)
from app.bootstrap import build_current_corpus_dense_evaluator
from app.config import load_settings
from app.infrastructure.retrieval.embedding_model_presets import (
    EMBEDDING_MODEL_PRESETS,
)
from app.infrastructure.retrieval.retrieval_evaluation_json import (
    load_retrieval_evaluation_dataset,
)
from app.infrastructure.retrieval.sentence_transformer_embedder import (
    SentenceTransformerTextEmbedder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current PostgreSQL corpus with a real "
            "Dense model and a versioned relevance dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/evaluation/huawei-campus-faq-retrieval-v1.json"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--preset",
        choices=tuple(EMBEDDING_MODEL_PRESETS),
        default="qwen3-embedding-0.6b",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=Path(".model-cache"),
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true")
    return parser.parse_args()


def report_to_payload(
    report: DenseRetrievalExperimentReport,
) -> dict[str, Any]:
    metrics = report.metrics
    return {
        "retriever": "dense",
        "dataset_id": report.dataset_id,
        "annotation_status": report.annotation_status,
        "corpus_size": report.corpus_size,
        "chunker_version": report.chunker_version,
        "embedding_identity": report.embedding_identity,
        "corpus_fingerprint": report.corpus_fingerprint,
        "top_k": metrics.top_k,
        "case_count": metrics.case_count,
        "macro_recall_at_k": round(metrics.macro_recall_at_k, 6),
        "mean_reciprocal_rank": round(
            metrics.mean_reciprocal_rank,
            6,
        ),
        "macro_ndcg_at_k": round(metrics.macro_ndcg_at_k, 6),
        "cases": [
            {
                "case_id": item.case_id,
                "recall_at_k": round(item.recall_at_k, 6),
                "reciprocal_rank": round(
                    item.reciprocal_rank,
                    6,
                ),
                "ndcg_at_k": round(item.ndcg_at_k, 6),
                "first_relevant_rank": item.first_relevant_rank,
                "retrieved_evidence_ids": list(
                    item.retrieved_evidence_ids
                ),
            }
            for item in metrics.cases
        ],
    }


def main() -> int:
    args = parse_args()
    dataset = load_retrieval_evaluation_dataset(args.dataset)
    embedder = SentenceTransformerTextEmbedder(
        EMBEDDING_MODEL_PRESETS[args.preset],
        device=args.device,
        batch_size=args.batch_size,
        cache_folder=args.cache_folder,
        local_files_only=not args.allow_download,
    )
    evaluator = build_current_corpus_dense_evaluator(
        load_settings(),
        embedder,
    )
    report = evaluator.evaluate(
        dataset,
        top_k=args.top_k,
        allow_provisional=args.allow_provisional,
    )
    print(
        json.dumps(
            report_to_payload(report),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
