from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.application.retrieval.retrieval_comparison import (
    RetrievalComparisonReport,
    compare_retrieval_reports,
)
from app.bootstrap import (
    build_current_corpus_bm25_evaluator,
    build_current_corpus_dense_evaluator,
)
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
            "Compare BM25 and Dense with the same corpus, dataset "
            "and Top-K cutoff."
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
    report: RetrievalComparisonReport,
) -> dict[str, Any]:
    return {
        "baseline": report.baseline_name,
        "candidate": report.candidate_name,
        "top_k": report.top_k,
        "case_count": report.case_count,
        "macro_recall_delta": round(report.macro_recall_delta, 6),
        "mean_reciprocal_rank_delta": round(
            report.mean_reciprocal_rank_delta,
            6,
        ),
        "macro_ndcg_delta": round(report.macro_ndcg_delta, 6),
        "cases": [
            {
                "case_id": item.case_id,
                "recall": {
                    "bm25": round(item.baseline_recall_at_k, 6),
                    "dense": round(item.candidate_recall_at_k, 6),
                    "delta": round(item.recall_delta, 6),
                },
                "reciprocal_rank": {
                    "bm25": round(
                        item.baseline_reciprocal_rank,
                        6,
                    ),
                    "dense": round(
                        item.candidate_reciprocal_rank,
                        6,
                    ),
                    "delta": round(item.reciprocal_rank_delta, 6),
                },
                "ndcg": {
                    "bm25": round(item.baseline_ndcg_at_k, 6),
                    "dense": round(item.candidate_ndcg_at_k, 6),
                    "delta": round(item.ndcg_delta, 6),
                },
            }
            for item in report.cases
        ],
    }


def main() -> int:
    args = parse_args()
    settings = load_settings()
    dataset = load_retrieval_evaluation_dataset(args.dataset)
    embedder = SentenceTransformerTextEmbedder(
        EMBEDDING_MODEL_PRESETS[args.preset],
        device=args.device,
        batch_size=args.batch_size,
        cache_folder=args.cache_folder,
        local_files_only=not args.allow_download,
    )
    bm25_report = build_current_corpus_bm25_evaluator(
        settings
    ).evaluate(
        dataset,
        top_k=args.top_k,
        allow_provisional=args.allow_provisional,
    )
    dense_report = build_current_corpus_dense_evaluator(
        settings,
        embedder,
    ).evaluate(
        dataset,
        top_k=args.top_k,
        allow_provisional=args.allow_provisional,
    )
    comparison = compare_retrieval_reports(
        "bm25",
        bm25_report.metrics,
        "dense",
        dense_report.metrics,
    )
    print(
        json.dumps(
            report_to_payload(comparison),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
