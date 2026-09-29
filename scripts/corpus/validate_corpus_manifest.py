from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.application.documents.corpus_profiling import build_corpus_profile
from app.infrastructure.documents.corpus_manifest_csv import (
    load_verified_corpus_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the JobScope corpus manifest and recheck every "
            "local artifact hash."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verified = load_verified_corpus_manifest(
        args.manifest,
        artifact_root=args.artifact_root,
    )
    profile = build_corpus_profile(verified)
    print(
        json.dumps(
            asdict(profile),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
