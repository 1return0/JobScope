from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.application.retrieval.current_corpus_dense_search import (
    CurrentCorpusDenseSearchReport,
)
from app.bootstrap import build_current_corpus_dense_search_service
from app.config import load_settings
from app.domain.documents.document_chunking import ChunkingConfig
from app.infrastructure.retrieval.embedding_model_presets import (
    EMBEDDING_MODEL_PRESETS,
)
from app.infrastructure.retrieval.sentence_transformer_embedder import (
    SentenceTransformerTextEmbedder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the current PostgreSQL corpus, build a real in-memory "
            "dense index and return semantic Top-K evidence."
        )
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-reference")
    parser.add_argument(
        "--chunker-version",
        default=ChunkingConfig().identity,
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
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow missing model files to be downloaded.",
    )
    return parser.parse_args()


def report_to_payload(
    report: CurrentCorpusDenseSearchReport,
) -> dict[str, Any]:
    return {
        "query": report.query,
        "top_k": report.top_k,
        "corpus_size": report.corpus_size,
        "chunker_version": report.chunker_version,
        "embedding_identity": report.embedding_identity,
        "corpus_fingerprint": report.corpus_fingerprint,
        "result_count": len(report.results),
        "results": [
            {
                "rank": item.rank,
                "score": round(item.score, 6),
                "evidence_id": item.chunk.evidence_id,
                "source_reference": item.chunk.source_reference,
                "source_fragment_ordinal": (
                    item.chunk.source_fragment_ordinal
                ),
                "chunk_ordinal": item.chunk.chunk_ordinal,
                "text": item.chunk.text,
                "location": {
                    "page_number": item.chunk.location.page_number,
                    "heading_path": list(
                        item.chunk.location.heading_path
                    ),
                    "sheet_name": item.chunk.location.sheet_name,
                    "cell_reference": (
                        item.chunk.location.cell_reference
                    ),
                    "slide_number": (
                        item.chunk.location.slide_number
                    ),
                },
            }
            for item in report.results
        ],
    }


def main() -> int:
    args = parse_args()
    embedder = SentenceTransformerTextEmbedder(
        EMBEDDING_MODEL_PRESETS[args.preset],
        device=args.device,
        batch_size=args.batch_size,
        cache_folder=args.cache_folder,
        local_files_only=not args.allow_download,
    )
    service = build_current_corpus_dense_search_service(
        load_settings(),
        embedder,
        chunker_version=args.chunker_version,
    )
    report = service.search(
        args.query,
        top_k=args.top_k,
        source_reference=args.source_reference,
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
