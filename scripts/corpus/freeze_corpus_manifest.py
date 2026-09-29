from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.infrastructure.documents.corpus_manifest_csv import (
    load_verified_corpus_manifest,
)
from app.infrastructure.documents.corpus_manifest_freeze import (
    freeze_verified_corpus_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze one immutable JobScope corpus manifest."
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
        "--output-directory",
        type=Path,
        default=Path("data/corpus-manifests"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verified = load_verified_corpus_manifest(
        args.manifest,
        artifact_root=args.artifact_root,
    )
    report = freeze_verified_corpus_manifest(
        verified,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "manifest_identity": report.manifest_identity,
                "path": str(report.path.resolve()),
                "created": report.created,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
