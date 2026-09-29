from __future__ import annotations

import argparse
import json
from typing import Any

from app.application.retrieval.current_corpus_search import (
    CurrentCorpusSearchReport,
)
from app.bootstrap import build_current_corpus_search_service
from app.config import load_settings
from app.domain.documents.document_chunking import ChunkingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the current JobScope corpus from PostgreSQL and "
            "run the in-memory BM25 lexical baseline."
        )
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-reference")
    parser.add_argument(
        "--chunker-version",
        default=ChunkingConfig().identity,
    )
    return parser.parse_args()


def report_to_payload(
    report: CurrentCorpusSearchReport,
) -> dict[str, Any]:
    return {
        "query": report.query,
        "top_k": report.top_k,
        "corpus_size": report.corpus_size,
        "chunker_version": report.chunker_version,
        "tokenizer_version": report.tokenizer_version,
        "result_count": len(report.results),
        "results": [
            {
                "rank": item.rank,
                "score": round(item.score, 6),
                "matched_terms": list(item.matched_terms),
                "evidence_id": item.chunk.evidence_id,
                "source_reference": (
                    item.chunk.source_reference
                ),
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
    service = build_current_corpus_search_service(
        load_settings(),
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
