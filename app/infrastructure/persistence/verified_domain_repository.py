from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.sources.source_trust import (
    VerifiedCompanyDomain,
    normalize_company,
    normalize_domain,
    verification_fingerprint,
)
from app.infrastructure.persistence.models import VerifiedCompanyDomainRow


@dataclass(frozen=True, slots=True)
class ImportReport:
    total: int
    inserted: int
    skipped: int
    dry_run: bool


class SqlAlchemyVerifiedDomainRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def add(self, record: VerifiedCompanyDomain) -> None:
        self.add_many([record])

    def preview_import(
        self,
        records: list[VerifiedCompanyDomain],
    ) -> ImportReport:
        with self._session_factory() as session:
            existing_fingerprints = self._existing_fingerprints(
                session
            )
        new_records = self._deduplicate_new_records(
            records,
            existing_fingerprints,
        )
        return ImportReport(
            total=len(records),
            inserted=len(new_records),
            skipped=len(records) - len(new_records),
            dry_run=True,
        )

    def add_many(
        self,
        records: list[VerifiedCompanyDomain],
    ) -> ImportReport:
        with self._session_factory.begin() as session:
            existing_fingerprints = self._existing_fingerprints(
                session
            )
            new_records = self._deduplicate_new_records(
                records,
                existing_fingerprints,
            )
            session.add_all(
                [self._to_row(record) for record in new_records]
            )

        return ImportReport(
            total=len(records),
            inserted=len(new_records),
            skipped=len(records) - len(new_records),
            dry_run=False,
        )

    def list_current_records(self) -> list[VerifiedCompanyDomain]:
        statement: Select = select(
            VerifiedCompanyDomainRow
        ).order_by(
            VerifiedCompanyDomainRow.normalized_company,
            VerifiedCompanyDomainRow.normalized_domain,
            VerifiedCompanyDomainRow.verified_at.desc(),
            VerifiedCompanyDomainRow.id.desc(),
        )

        with self._session_factory() as session:
            rows = session.scalars(statement).all()

        current_records: list[VerifiedCompanyDomain] = []
        seen_keys: set[tuple[str, str]] = set()
        for row in rows:
            key = (row.normalized_company, row.normalized_domain)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            current_records.append(
                VerifiedCompanyDomain(
                    company=row.company,
                    domain=row.domain,
                    evidence_reference=row.evidence_reference,
                    verified_at=row.verified_at,
                    verified_by=row.verified_by,
                    status=row.status,
                )
            )
        return current_records

    @staticmethod
    def _existing_fingerprints(session: Session) -> set[str]:
        return set(
            session.scalars(
                select(
                    VerifiedCompanyDomainRow.record_fingerprint
                )
            ).all()
        )

    @staticmethod
    def _deduplicate_new_records(
        records: list[VerifiedCompanyDomain],
        existing_fingerprints: set[str],
    ) -> list[VerifiedCompanyDomain]:
        new_records: list[VerifiedCompanyDomain] = []
        seen_fingerprints = set(existing_fingerprints)
        for record in records:
            fingerprint = verification_fingerprint(record)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            new_records.append(record)
        return new_records

    @staticmethod
    def _to_row(
        record: VerifiedCompanyDomain,
    ) -> VerifiedCompanyDomainRow:
        return VerifiedCompanyDomainRow(
            record_fingerprint=verification_fingerprint(record),
            company=record.company.strip(),
            normalized_company=normalize_company(record.company),
            domain=normalize_domain(record.domain),
            normalized_domain=normalize_domain(record.domain),
            evidence_reference=record.evidence_reference.strip(),
            verified_at=record.verified_at,
            verified_by=record.verified_by.strip(),
            status=record.status,
        )
