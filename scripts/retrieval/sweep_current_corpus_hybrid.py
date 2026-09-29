from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.bootstrap import build_current_corpus_hybrid_evaluator
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


def _positive_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(","))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Hybrid RRF parameters while reusing one Dense index."
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
        "--candidate-ks",
        type=_positive_int_list,
        default=(5, 10, 20),
    )
    parser.add_argument(
        "--rank-constants",
        type=_positive_int_list,
        default=(10, 30, 60),
    )
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
    report = build_current_corpus_hybrid_evaluator(
        load_settings(),
        embedder,
    ).evaluate_parameter_grid(
        dataset,
        top_k=args.top_k,
        candidate_ks=args.candidate_ks,
        rank_constants=args.rank_constants,
        allow_provisional=args.allow_provisional,
    )
    payload = {
        "dataset_id": report.dataset_id,
        "annotation_status": report.annotation_status,
        "corpus_size": report.corpus_size,
        "chunker_version": report.chunker_version,
        "tokenizer_version": report.tokenizer_version,
        "embedding_identity": report.embedding_identity,
        "corpus_fingerprint": report.corpus_fingerprint,
        "evaluations": [
            {
                "candidate_k": item.candidate_k,
                "rank_constant": item.rank_constant,
                "top_k": item.metrics.top_k,
                "macro_recall_at_k": round(
                    item.metrics.macro_recall_at_k,
                    6,
                ),
                "mean_reciprocal_rank": round(
                    item.metrics.mean_reciprocal_rank,
                    6,
                ),
                "macro_ndcg_at_k": round(
                    item.metrics.macro_ndcg_at_k,
                    6,
                ),
            }
            for item in report.evaluations
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
