import unittest
from datetime import date

from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from app.domain.sources.source_trust import (
    VerifiedCompanyDomain,
    VerifiedDomainRegistry,
)
from app.infrastructure.persistence.database import (
    Base,
    create_session_factory,
)
from app.infrastructure.persistence.models import VerifiedCompanyDomainRow
from app.infrastructure.persistence.verified_domain_repository import (
    SqlAlchemyVerifiedDomainRepository,
)


class SqlAlchemyVerifiedDomainRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.repository = SqlAlchemyVerifiedDomainRepository(
            self.session_factory
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_round_trips_verification_evidence(self) -> None:
        self.repository.add(
            VerifiedCompanyDomain(
                company="Example Technology",
                domain="Example.COM.",
                evidence_reference="manual-evidence-001",
                verified_at=date(2026, 7, 25),
                verified_by="reviewer-a",
                status="active",
            )
        )

        records = self.repository.list_current_records()

        self.assertEqual(1, len(records))
        self.assertEqual("example.com", records[0].domain)
        self.assertEqual(
            "manual-evidence-001",
            records[0].evidence_reference,
        )

    def test_latest_revocation_prevents_domain_matching(self) -> None:
        self.repository.add(
            VerifiedCompanyDomain(
                company="Example Technology",
                domain="example.com",
                evidence_reference="verification-001",
                verified_at=date(2026, 7, 20),
                verified_by="reviewer-a",
                status="active",
            )
        )
        self.repository.add(
            VerifiedCompanyDomain(
                company="Example Technology",
                domain="example.com",
                evidence_reference="revocation-001",
                verified_at=date(2026, 7, 25),
                verified_by="reviewer-b",
                status="revoked",
            )
        )

        registry = VerifiedDomainRegistry(
            self.repository.list_current_records()
        )

        self.assertIsNone(
            registry.find_matching_domain(
                "Example Technology",
                "careers.example.com",
            )
        )

    def test_repeated_import_is_idempotent(self) -> None:
        record = VerifiedCompanyDomain(
            company="Example Technology",
            domain="example.com",
            evidence_reference="verification-001",
            verified_at=date(2026, 7, 25),
            verified_by="reviewer-a",
            status="active",
        )

        preview = self.repository.preview_import([record, record])
        first_import = self.repository.add_many([record, record])
        second_import = self.repository.add_many([record])
        records = self.repository.list_current_records()

        self.assertTrue(preview.dry_run)
        self.assertEqual(1, preview.inserted)
        self.assertEqual(1, preview.skipped)
        self.assertEqual(1, first_import.inserted)
        self.assertEqual(1, first_import.skipped)
        self.assertEqual(0, second_import.inserted)
        self.assertEqual(1, second_import.skipped)
        self.assertEqual(1, len(records))

    def test_transaction_rolls_back_when_later_insert_fails(
        self,
    ) -> None:
        first_row = self._build_row(
            fingerprint="same-fingerprint",
            domain="first.example",
        )
        conflicting_row = self._build_row(
            fingerprint="same-fingerprint",
            domain="second.example",
        )

        with self.assertRaises(IntegrityError):
            with self.session_factory.begin() as session:
                session.add(first_row)
                session.flush()
                session.add(conflicting_row)
                session.flush()

        with self.session_factory() as session:
            row_count = session.scalar(
                select(func.count()).select_from(
                    VerifiedCompanyDomainRow
                )
            )

        self.assertEqual(0, row_count)

    @staticmethod
    def _build_row(
        fingerprint: str,
        domain: str,
    ) -> VerifiedCompanyDomainRow:
        return VerifiedCompanyDomainRow(
            record_fingerprint=fingerprint,
            company="Example Technology",
            normalized_company="example technology",
            domain=domain,
            normalized_domain=domain,
            evidence_reference="rollback-test-evidence",
            verified_at=date(2026, 7, 25),
            verified_by="test-reviewer",
            status="active",
        )


if __name__ == "__main__":
    unittest.main()
