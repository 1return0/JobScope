from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.application.documents.corpus_source_audit import audit_corpus_sources
from app.config import PROJECT_ROOT
from app.domain.sources.source_trust import VerifiedDomainRegistry
from app.infrastructure.documents.corpus_manifest_csv import (
    load_verified_corpus_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit official-company claims in a verified Corpus "
            "Manifest against the evidence-backed domain registry."
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
        "--verified-domains",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "verified-company-domains.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verified = load_verified_corpus_manifest(
        args.manifest,
        artifact_root=args.artifact_root,
    )
    registry = VerifiedDomainRegistry.from_csv(
        args.verified_domains
    )
    report = audit_corpus_sources(verified, registry)
    print(
        json.dumps(
            asdict(report),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.unverified_company_sources == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
