from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.application.retrieval.embedding_smoke_test import (
    EmbeddingSmokeCase,
    EmbeddingSmokeReport,
    EmbeddingSmokeTestService,
)
from app.infrastructure.retrieval.embedding_model_presets import (
    EMBEDDING_MODEL_PRESETS,
)
from app.infrastructure.retrieval.sentence_transformer_embedder import (
    SentenceTransformerTextEmbedder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one real local embedding model and run an isolated "
            "semantic smoke test without PostgreSQL."
        )
    )
    parser.add_argument(
        "--preset",
        choices=tuple(EMBEDDING_MODEL_PRESETS),
        default="qwen3-embedding-0.6b",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=Path(".model-cache"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--query",
        default="学校名单里没有我的大学怎么办？",
    )
    parser.add_argument(
        "--positive-document",
        default="列表中没有院校时，请手动输入标准院校名称。",
    )
    parser.add_argument(
        "--negative-document",
        default="面试结果会通过短信和邮件通知。",
    )
    return parser.parse_args()


def report_to_payload(report: EmbeddingSmokeReport) -> dict[str, Any]:
    return {
        "provider": report.provider,
        "model_name": report.model_name,
        "model_revision": report.model_revision,
        "embedding_identity": report.embedding_identity,
        "dimension": report.dimension,
        "query": report.query,
        "candidates": [
            {
                "label": "positive",
                "text": report.positive_document,
                "similarity": round(report.positive_similarity, 6),
            },
            {
                "label": "negative",
                "text": report.negative_document,
                "similarity": round(report.negative_similarity, 6),
            },
        ],
        "similarity_margin": round(report.similarity_margin, 6),
        "semantic_order_passed": report.semantic_order_passed,
        "scope": (
            "model-only smoke test; not a retrieval quality benchmark"
        ),
    }


def main() -> int:
    args = parse_args()
    embedder = SentenceTransformerTextEmbedder(
        EMBEDDING_MODEL_PRESETS[args.preset],
        device=args.device,
        batch_size=args.batch_size,
        cache_folder=args.cache_folder,
        local_files_only=args.local_files_only,
    )
    report = EmbeddingSmokeTestService(embedder).run(
        EmbeddingSmokeCase(
            query=args.query,
            positive_document=args.positive_document,
            negative_document=args.negative_document,
        )
    )
    print(
        json.dumps(
            report_to_payload(report),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.semantic_order_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
