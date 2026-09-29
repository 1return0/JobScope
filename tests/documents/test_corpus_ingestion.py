import unittest
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.application.documents.corpus_ingestion import (
    CorpusIngestionService,
    EmptyCorpusIngestionError,
)
from app.application.documents.document_parsing import (
    DocumentParserRegistry,
    DocumentParsingService,
)
from app.application.documents.document_processing import (
    DocumentProcessingService,
)
from app.domain.documents.corpus_manifest import (
    CorpusManifest,
    CorpusManifestEntry,
    VerifiedCorpusDocument,
    VerifiedCorpusManifest,
)
from app.domain.documents.document_chunking import StructureAwareChunker
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    EvidenceLocation,
    NoExtractableContentError,
    ParsedDocument,
    ParsedFragment,
)


class _RoutingHtmlParser:
    document_format = "html"

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        if "bad" in artifact.source_reference:
            raise NoExtractableContentError("document is empty")
        return ParsedDocument(
            artifact=artifact,
            fragments=(
                ParsedFragment(
                    ordinal=0,
                    text="Backend role requires Java and PostgreSQL.",
                    location=EvidenceLocation(
                        heading_path=("Backend Engineer",)
                    ),
                ),
            ),
        )


class _RecordingCorpusWriter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[DocumentArtifact, list, datetime]] = []

    def persist(
        self,
        artifact: DocumentArtifact,
        chunks: list,
        *,
        observed_at: datetime,
    ) -> object:
        if self.fail:
            raise RuntimeError("database unavailable")
        self.calls.append((artifact, chunks, observed_at))
        return _PersistReport.for_chunks(len(chunks))


@dataclass(frozen=True)
class _SnapshotReport:
    created: bool


@dataclass(frozen=True)
class _CountReport:
    total: int
    inserted: int
    skipped: int


@dataclass(frozen=True)
class _PersistReport:
    snapshot: _SnapshotReport
    chunks: _CountReport
    links: _CountReport

    @classmethod
    def for_chunks(cls, chunk_count: int) -> "_PersistReport":
        counts = _CountReport(
            total=chunk_count,
            inserted=chunk_count,
            skipped=0,
        )
        return cls(
            snapshot=_SnapshotReport(created=True),
            chunks=counts,
            links=counts,
        )


class CorpusIngestionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        parsing_service = DocumentParsingService(
            DocumentParserRegistry([_RoutingHtmlParser()])
        )
        self.processing_service = DocumentProcessingService(
            parsing_service,
            StructureAwareChunker(),
        )

    def test_continue_policy_persists_other_documents(self) -> None:
        writer = _RecordingCorpusWriter()
        service = CorpusIngestionService(
            self.processing_service,
            writer,
        )
        verified = self._manifest("bad-source", "good-source")

        report = service.ingest(
            verified,
            failure_policy="continue",
        )

        self.assertEqual(2, report.total)
        self.assertEqual(1, report.succeeded)
        self.assertEqual(1, report.parsing_failed)
        self.assertEqual(0, report.not_attempted)
        self.assertEqual(
            ("parsing_failed", "succeeded"),
            tuple(item.status for item in report.items),
        )
        self.assertEqual(
            "no-extractable-content",
            report.items[0].failure_code,
        )
        self.assertEqual(1, len(writer.calls))
        self.assertTrue(
            report.items[1].persistence.snapshot_created
        )
        self.assertEqual(
            report.items[1].chunk_count,
            report.items[1].persistence.chunks_inserted,
        )
        self.assertEqual(
            verified.documents[1].entry.captured_at,
            writer.calls[0][2],
        )

    def test_stop_policy_marks_remaining_documents_not_attempted(
        self,
    ) -> None:
        writer = _RecordingCorpusWriter()
        service = CorpusIngestionService(
            self.processing_service,
            writer,
        )
        verified = self._manifest("bad-source", "good-source")

        report = service.ingest(verified, failure_policy="stop")

        self.assertEqual(
            ("parsing_failed", "not_attempted"),
            tuple(item.status for item in report.items),
        )
        self.assertEqual(0, report.succeeded)
        self.assertEqual(1, report.parsing_failed)
        self.assertEqual(1, report.not_attempted)
        self.assertEqual([], writer.calls)

    def test_database_error_is_not_disguised_as_bad_document(self) -> None:
        service = CorpusIngestionService(
            self.processing_service,
            _RecordingCorpusWriter(fail=True),
        )

        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            service.ingest(self._manifest("good-source"))

    def test_rejects_empty_manifest(self) -> None:
        service = CorpusIngestionService(
            self.processing_service,
            _RecordingCorpusWriter(),
        )
        empty = VerifiedCorpusManifest(
            manifest=CorpusManifest(()),
            documents=(),
        )

        with self.assertRaisesRegex(
            EmptyCorpusIngestionError,
            "empty",
        ):
            service.ingest(empty)

    @staticmethod
    def _manifest(*source_ids: str) -> VerifiedCorpusManifest:
        entries: list[CorpusManifestEntry] = []
        documents: list[VerifiedCorpusDocument] = []
        for index, source_id in enumerate(source_ids):
            digest = f"{index + 1:064x}"
            final_url = f"https://careers.example/{source_id}.html"
            entry = CorpusManifestEntry(
                source_id=source_id,
                company="Example Technology",
                job_title="Backend Engineer",
                job_id=None,
                recruitment_type="campus",
                published_date=None,
                location="Shanghai",
                source_kind="official_company",
                source_url=final_url,
                final_url=final_url,
                captured_at=datetime(
                    2026,
                    8,
                    1,
                    10 + index,
                    tzinfo=UTC,
                ),
                artifact_path=Path(f"{source_id}.html"),
                content_sha256=digest,
                status="active",
            )
            artifact = DocumentArtifact(
                path=Path(f"C:/{source_id}.html"),
                source_reference=final_url,
                document_format="html",
                content_sha256=digest,
                byte_size=10,
            )
            entries.append(entry)
            documents.append(
                VerifiedCorpusDocument(
                    entry=entry,
                    artifact=artifact,
                )
            )

        return VerifiedCorpusManifest(
            manifest=CorpusManifest(tuple(entries)),
            documents=tuple(documents),
        )


if __name__ == "__main__":
    unittest.main()
