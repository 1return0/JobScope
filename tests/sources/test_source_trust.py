import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.sources.source_trust import (
    VerifiedCompanyDomain,
    VerifiedDomainRegistry,
    load_verified_domain_records,
)


class VerifiedDomainRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = VerifiedDomainRegistry(
            [
                VerifiedCompanyDomain(
                    company="Example Technology",
                    domain="example.com",
                    evidence_reference="manual-test-evidence",
                    verified_at=date(2026, 7, 25),
                    verified_by="test-reviewer",
                    status="active",
                )
            ]
        )

    def test_matches_exact_domain_and_subdomain(self) -> None:
        self.assertEqual(
            "example.com",
            self.registry.find_matching_domain(
                "Example Technology",
                "example.com",
            ),
        )
        self.assertEqual(
            "example.com",
            self.registry.find_matching_domain(
                "Example Technology",
                "careers.example.com",
            ),
        )

    def test_rejects_lookalike_domain(self) -> None:
        self.assertIsNone(
            self.registry.find_matching_domain(
                "Example Technology",
                "careers-example.com",
            )
        )
        self.assertIsNone(
            self.registry.find_matching_domain(
                "Example Technology",
                "example.com.evil.test",
            )
        )

    def test_does_not_share_domains_between_companies(self) -> None:
        self.assertIsNone(
            self.registry.find_matching_domain(
                "Different Company",
                "careers.example.com",
            )
        )

    def test_normalizes_company_and_domain_before_matching(self) -> None:
        registry = VerifiedDomainRegistry(
            [
                VerifiedCompanyDomain(
                    company="  Example TECHNOLOGY  ",
                    domain="Example.COM.",
                    evidence_reference="manual-test-evidence",
                    verified_at=date(2026, 7, 25),
                    verified_by="test-reviewer",
                    status="active",
                )
            ]
        )

        self.assertEqual(
            "example.com",
            registry.find_matching_domain(
                "example technology",
                "CAREERS.EXAMPLE.COM.",
            ),
        )

    def test_csv_loader_reads_active_and_revoked_records(self) -> None:
        csv_content = (
            "company,domain,evidence_reference,verified_at,"
            "verified_by,status\n"
            "Example Technology,example.com,evidence-1,"
            "2026-07-25,reviewer-a,active\n"
            "Old Company,old.example,evidence-2,"
            "2026-07-20,reviewer-b,revoked\n"
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "verified-domains.csv"
            path.write_text(csv_content, encoding="utf-8")
            records = load_verified_domain_records(path)
            registry = VerifiedDomainRegistry(records)

        self.assertEqual(2, len(records))
        self.assertEqual(
            "example.com",
            registry.find_matching_domain(
                "Example Technology",
                "careers.example.com",
            ),
        )
        self.assertIsNone(
            registry.find_matching_domain(
                "Old Company",
                "old.example",
            )
        )

    def test_csv_loader_rejects_missing_required_fields(self) -> None:
        csv_content = "company,domain\nExample,example.com\n"

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "verified-domains.csv"
            path.write_text(csv_content, encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "missing fields",
            ):
                load_verified_domain_records(path)

    def test_csv_loader_reports_invalid_row_number(self) -> None:
        csv_content = (
            "company,domain,evidence_reference,verified_at,"
            "verified_by,status\n"
            "Example Technology,https://example.com,evidence-1,"
            "not-a-date,reviewer-a,unknown\n"
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "verified-domains.csv"
            path.write_text(csv_content, encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "row 2",
            ):
                load_verified_domain_records(path)


if __name__ == "__main__":
    unittest.main()
