import unittest
from datetime import UTC, date, datetime
from pathlib import Path

from app.application.documents.corpus_profiling import build_corpus_profile
from app.domain.documents.corpus_manifest import (
    CorpusManifest,
    CorpusManifestEntry,
    VerifiedCorpusDocument,
    VerifiedCorpusManifest,
)
from app.domain.documents.document_ingestion import DocumentArtifact


class CorpusProfilingTest(unittest.TestCase):
    def test_reports_coverage_freshness_and_duplicate_content(self) -> None:
        first = self._document(
            source_id="company-a-backend",
            company="Company A",
            document_format="html",
            content_hash="a" * 64,
            byte_size=100,
            source_kind="official_company",
            status="active",
            published_date=date(2026, 7, 1),
        )
        second = self._document(
            source_id="company-a-data",
            company="Company A",
            document_format="pdf",
            content_hash="b" * 64,
            byte_size=300,
            source_kind="official_company",
            status="unknown",
            published_date=None,
        )
        third = self._document(
            source_id="university-a-notice",
            company="Company B",
            document_format="pdf",
            content_hash="a" * 64,
            byte_size=100,
            source_kind="official_university",
            status="active",
            published_date=None,
        )
        documents = (first, second, third)
        verified = VerifiedCorpusManifest(
            manifest=CorpusManifest(
                tuple(document.entry for document in documents)
            ),
            documents=documents,
        )

        profile = build_corpus_profile(verified)

        self.assertEqual(3, profile.document_count)
        self.assertEqual(2, profile.company_count)
        self.assertEqual(500, profile.total_bytes)
        self.assertEqual(2, profile.unique_content_count)
        self.assertEqual(1, profile.duplicate_content_group_count)
        self.assertEqual(2, profile.duplicate_content_document_count)
        self.assertEqual(2, profile.missing_published_date_count)
        self.assertEqual(
            {"Company A": 2, "Company B": 1},
            profile.company_counts,
        )
        self.assertEqual({"html": 1, "pdf": 2}, profile.format_counts)
        self.assertEqual(
            {"official_company": 2, "official_university": 1},
            profile.source_kind_counts,
        )
        self.assertEqual(
            {"active": 2, "unknown": 1},
            profile.status_counts,
        )

    @staticmethod
    def _document(
        *,
        source_id: str,
        company: str,
        document_format,
        content_hash: str,
        byte_size: int,
        source_kind,
        status,
        published_date: date | None,
    ) -> VerifiedCorpusDocument:
        source_url = f"https://careers.example/{source_id}"
        entry = CorpusManifestEntry(
            source_id=source_id,
            company=company,
            job_title="Backend Engineer",
            job_id=None,
            recruitment_type="campus",
            published_date=published_date,
            location="Shanghai",
            source_kind=source_kind,
            source_url=source_url,
            final_url=source_url,
            captured_at=datetime(2026, 8, 1, tzinfo=UTC),
            artifact_path=Path(f"{source_id}.{document_format}"),
            content_sha256=content_hash,
            status=status,
        )
        artifact = DocumentArtifact(
            path=Path(f"C:/{source_id}.{document_format}"),
            source_reference=source_url,
            document_format=document_format,
            content_sha256=content_hash,
            byte_size=byte_size,
        )
        return VerifiedCorpusDocument(entry=entry, artifact=artifact)


if __name__ == "__main__":
    unittest.main()
