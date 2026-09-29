from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

from app.infrastructure.documents.corpus_manifest_csv import (
    register_corpus_artifact,
)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "date must use YYYY-MM-DD"
        ) from error


def _parse_datetime(value: str) -> datetime:
    normalized = (
        f"{value[:-1]}+00:00" if value.endswith("Z") else value
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "datetime must use ISO 8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError(
            "datetime must include a timezone"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect one local recruitment artifact and create or "
            "replace a validated JobScope corpus manifest row."
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
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--job-title", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--recruitment-type", required=True)
    parser.add_argument("--published-date", type=_parse_date)
    parser.add_argument("--location", required=True)
    parser.add_argument(
        "--source-kind",
        choices=(
            "official_company",
            "official_university",
            "job_board",
        ),
        required=True,
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--final-url", required=True)
    parser.add_argument(
        "--captured-at",
        type=_parse_datetime,
        default=datetime.now(UTC),
    )
    parser.add_argument(
        "--status",
        choices=("active", "expired", "unknown"),
        default="unknown",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "replace the existing row with the same source_id after "
            "validating stable source identity"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = register_corpus_artifact(
        args.manifest,
        artifact_root=args.artifact_root,
        artifact_path=args.artifact_path,
        source_id=args.source_id,
        company=args.company,
        job_title=args.job_title,
        job_id=args.job_id,
        recruitment_type=args.recruitment_type,
        published_date=args.published_date,
        location=args.location,
        source_kind=args.source_kind,
        source_url=args.source_url,
        final_url=args.final_url,
        captured_at=args.captured_at,
        status=args.status,
        notes=args.notes,
        replace_existing=args.replace_existing,
    )
    print(
        json.dumps(
            {
                "source_id": report.entry.source_id,
                "artifact_path": report.entry.artifact_path.as_posix(),
                "document_format": report.document_format,
                "byte_size": report.byte_size,
                "content_sha256": report.entry.content_sha256,
                "manifest_identity": report.manifest_identity,
                "document_count": report.document_count,
                "operation": report.operation,
                "content_changed": report.content_changed,
                "previous_content_sha256": (
                    report.previous_content_sha256
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
