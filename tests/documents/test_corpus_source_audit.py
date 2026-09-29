import unittest
from datetime import UTC, date, datetime
from pathlib import Path

from app.application.documents.corpus_source_audit import audit_corpus_sources
from app.domain.documents.corpus_manifest import (
    CorpusManifest,
    CorpusManifestEntry,
    VerifiedCorpusDocument,
    VerifiedCorpusManifest,
)
from app.domain.documents.document_ingestion import DocumentArtifact
from app.domain.sources.source_trust import (
    VerifiedCompanyDomain,
    VerifiedDomainRegistry,
)


class CorpusSourceAuditTest(unittest.TestCase):
    def test_distinguishes_verified_unverified_and_non_company_sources(
        self,
    ) -> None:
        documents = (
            self._document(
                source_id="company-a-job",
                company="Company A",
                source_kind="official_company",
                final_url="https://jobs.company-a.example/backend",
            ),
            self._document(
                source_id="company-b-job",
                company="Company B",
                source_kind="official_company",
                final_url="https://careers.company-b.example/backend",
            ),
            self._document(
                source_id="university-notice",
                company="Company A",
                source_kind="official_university",
                final_url="https://jobs.university.example/notice",
            ),
        )
        verified = VerifiedCorpusManifest(
            manifest=CorpusManifest(
                tuple(document.entry for document in documents)
            ),
            documents=documents,
        )
        registry = VerifiedDomainRegistry(
            [
                VerifiedCompanyDomain(
                    company="Company A",
                    domain="company-a.example",
                    evidence_reference="manual-evidence",
                    verified_at=date(2026, 8, 2),
                    verified_by="reviewer",
                    status="active",
                )
            ]
        )

        report = audit_corpus_sources(verified, registry)

        self.assertEqual(3, report.total)
        self.assertEqual(2, report.official_company_claims)
        self.assertEqual(1, report.verified_company_sources)
        self.assertEqual(1, report.unverified_company_sources)
        self.assertEqual(1, report.not_applicable)
        self.assertEqual(
            (
                "verified_company_domain",
                "unverified_company_domain",
                "not_applicable",
            ),
            tuple(item.audit_status for item in report.items),
        )
        self.assertEqual(
            "company-a.example",
            report.items[0].matched_domain,
        )
        self.assertIsNone(report.items[1].evidence_reference)

    @staticmethod
    def _document(
        *,
        source_id: str,
        company: str,
        source_kind,
        final_url: str,
    ) -> VerifiedCorpusDocument:
        content_hash = source_id.encode().hex().ljust(64, "0")[:64]
        entry = CorpusManifestEntry(
            source_id=source_id,
            company=company,
            job_title="Backend Engineer",
            job_id=None,
            recruitment_type="campus",
            published_date=None,
            location="Shanghai",
            source_kind=source_kind,
            source_url=final_url,
            final_url=final_url,
            captured_at=datetime(2026, 8, 2, tzinfo=UTC),
            artifact_path=Path(f"{source_id}.html"),
            content_sha256=content_hash,
            status="unknown",
        )
        artifact = DocumentArtifact(
            path=Path(f"C:/{source_id}.html"),
            source_reference=final_url,
            document_format="html",
            content_sha256=content_hash,
            byte_size=100,
        )
        return VerifiedCorpusDocument(entry=entry, artifact=artifact)


if __name__ == "__main__":
    unittest.main()
