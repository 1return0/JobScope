from __future__ import annotations

import argparse
from pathlib import Path

from app.config import load_settings
from app.domain.sources.source_trust import load_verified_domain_records
from app.infrastructure.persistence.database import (
    create_postgresql_engine,
    create_session_factory,
)
from app.infrastructure.persistence.verified_domain_repository import (
    SqlAlchemyVerifiedDomainRepository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or apply verified company domain CSV imports."
        )
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="CSV path; defaults to the configured verified-domain file.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write new records. Without this flag, only preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    csv_path = args.file or settings.verified_domains_path
    records = load_verified_domain_records(csv_path)

    engine = create_postgresql_engine(settings)
    try:
        repository = SqlAlchemyVerifiedDomainRepository(
            create_session_factory(engine)
        )
        if args.apply:
            report = repository.add_many(records)
        else:
            report = repository.preview_import(records)
    finally:
        engine.dispose()

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(
        f"mode={mode} total={report.total} "
        f"inserted={report.inserted} skipped={report.skipped}"
    )


if __name__ == "__main__":
    main()
