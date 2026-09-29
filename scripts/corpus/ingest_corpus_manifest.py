from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.application.documents.corpus_ingestion import CorpusIngestionReport
from app.bootstrap import build_corpus_ingestion_service
from app.config import load_settings
from app.infrastructure.documents.corpus_manifest_csv import (
    load_verified_corpus_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a JobScope corpus manifest, process every document "
            "and persist the resulting corpus to PostgreSQL."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/job-sources.csv"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("data/corpus-artifacts"),
    )
    parser.add_argument(
        "--failure-policy",
        choices=("continue", "stop"),
        default="continue",
    )
    return parser.parse_args()


def report_to_payload(
    report: CorpusIngestionReport,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in report.items:
        persistence = item.persistence
        items.append(
            {
                "source_id": item.source_id,
                "status": item.status,
                "chunk_count": item.chunk_count,
                "failure_code": item.failure_code,
                "persistence": (
                    {
                        "snapshot_created": (
                            persistence.snapshot_created
                        ),
                        "chunks_inserted": (
                            persistence.chunks_inserted
                        ),
                        "chunks_skipped": (
                            persistence.chunks_skipped
                        ),
                        "links_inserted": (
                            persistence.links_inserted
                        ),
                        "links_skipped": (
                            persistence.links_skipped
                        ),
                    }
                    if persistence is not None
                    else None
                ),
            }
        )

    return {
        "manifest_identity": report.manifest_identity,
        "failure_policy": report.failure_policy,
        "total": report.total,
        "succeeded": report.succeeded,
        "parsing_failed": report.parsing_failed,
        "not_attempted": report.not_attempted,
        "items": items,
    }


def report_exit_code(report: CorpusIngestionReport) -> int:
    return 0 if report.parsing_failed == 0 else 2


def main() -> int:
    args = parse_args()
    verified = load_verified_corpus_manifest(
        args.manifest,
        artifact_root=args.artifact_root,
    )
    ingestion_service = build_corpus_ingestion_service(
        load_settings()
    )
    report = ingestion_service.ingest(
        verified,
        failure_policy=args.failure_policy,
    )
    print(
        json.dumps(
            report_to_payload(report),
            ensure_ascii=False,
            indent=2,
        )
    )
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
