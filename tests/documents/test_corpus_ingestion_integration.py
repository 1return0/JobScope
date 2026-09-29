import csv
import hashlib
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from app.application.documents.corpus_ingestion import CorpusIngestionService
from app.bootstrap import build_document_processing_service
from app.infrastructure.documents.corpus_manifest_csv import (
    MANIFEST_COLUMNS,
    load_verified_corpus_manifest,
    register_corpus_artifact,
)
from app.infrastructure.persistence.database import Base, create_session_factory
from app.infrastructure.persistence.document_corpus_repository import (
    SqlAlchemyDocumentCorpusRepository,
)
from app.infrastructure.persistence.models import (
    DocumentSnapshotChunkRow,
    DocumentSnapshotRow,
    EvidenceChunkRow,
)


class CorpusIngestionIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.repository = SqlAlchemyDocumentCorpusRepository(
            self.session_factory
        )
        self.service = CorpusIngestionService(
            build_document_processing_service(),
            self.repository,
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_manifest_ingestion_writes_three_tables_idempotently(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            verified = self._verified_html_manifest(Path(directory))

            first = self.service.ingest(verified)
            first_counts = self._row_counts()
            repeated = self.service.ingest(verified)
            repeated_counts = self._row_counts()

        first_item = first.items[0]
        repeated_item = repeated.items[0]
        self.assertEqual("succeeded", first_item.status)
        self.assertTrue(first_item.persistence.snapshot_created)
        self.assertEqual(
            first_item.chunk_count,
            first_item.persistence.chunks_inserted,
        )
        self.assertEqual(
            first_item.chunk_count,
            first_item.persistence.links_inserted,
        )
        self.assertEqual(
            (1, first_item.chunk_count, first_item.chunk_count),
            first_counts,
        )

        self.assertFalse(repeated_item.persistence.snapshot_created)
        self.assertEqual(0, repeated_item.persistence.chunks_inserted)
        self.assertEqual(
            repeated_item.chunk_count,
            repeated_item.persistence.chunks_skipped,
        )
        self.assertEqual(0, repeated_item.persistence.links_inserted)
        self.assertEqual(
            repeated_item.chunk_count,
            repeated_item.persistence.links_skipped,
        )
        self.assertEqual(first_counts, repeated_counts)

    def test_registered_content_update_creates_next_snapshot(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "artifacts"
            artifact_root.mkdir()
            manifest_path = root / "manifest.csv"
            with manifest_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as manifest_file:
                csv.DictWriter(
                    manifest_file,
                    fieldnames=MANIFEST_COLUMNS,
                ).writeheader()

            first_path = Path("example/backend-v1.html")
            second_path = Path("example/backend-v2.html")
            (artifact_root / "example").mkdir()
            (artifact_root / first_path).write_text(
                "<h1>Backend</h1><p>Java required.</p>",
                encoding="utf-8",
            )
            self._register_artifact(
                manifest_path,
                artifact_root,
                first_path,
                captured_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
            first_verified = load_verified_corpus_manifest(
                manifest_path,
                artifact_root=artifact_root,
            )
            self.service.ingest(first_verified)

            (artifact_root / second_path).write_text(
                "<h1>Backend</h1><p>Java and PostgreSQL required.</p>",
                encoding="utf-8",
            )
            self._register_artifact(
                manifest_path,
                artifact_root,
                second_path,
                captured_at=datetime(2026, 8, 2, tzinfo=UTC),
                replace_existing=True,
            )
            second_verified = load_verified_corpus_manifest(
                manifest_path,
                artifact_root=artifact_root,
            )
            second_report = self.service.ingest(second_verified)

        with self.session_factory() as session:
            snapshots = session.scalars(
                select(DocumentSnapshotRow).order_by(
                    DocumentSnapshotRow.version_number
                )
            ).all()

        self.assertEqual(2, len(snapshots))
        self.assertFalse(snapshots[0].is_current)
        self.assertTrue(snapshots[1].is_current)
        self.assertEqual((1, 2), tuple(
            snapshot.version_number for snapshot in snapshots
        ))
        self.assertTrue(
            second_report.items[0].persistence.snapshot_created
        )

    def _verified_html_manifest(self, root: Path):
        content = (
            b"<h1>Backend Engineer</h1>"
            b"<p>Requires Java and PostgreSQL.</p>"
        )
        content_sha256 = hashlib.sha256(content).hexdigest()
        artifact_root = root / "artifacts"
        artifact_path = artifact_root / "example" / "backend.html"
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_bytes(content)

        manifest_path = root / "manifest.csv"
        row = {
            "source_id": "example-2027-backend",
            "company": "Example Technology",
            "job_title": "Backend Engineer",
            "job_id": "backend-01",
            "recruitment_type": "campus",
            "published_date": "",
            "location": "Shanghai",
            "source_kind": "official_company",
            "source_url": "https://careers.example/jobs/backend",
            "final_url": "https://careers.example/jobs/backend.html",
            "captured_at": "2026-08-01T10:00:00+08:00",
            "artifact_path": "example/backend.html",
            "content_sha256": content_sha256,
            "status": "active",
            "notes": "",
        }
        with manifest_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=MANIFEST_COLUMNS,
            )
            writer.writeheader()
            writer.writerow(row)

        return load_verified_corpus_manifest(
            manifest_path,
            artifact_root=artifact_root,
        )

    def _row_counts(self) -> tuple[int, int, int]:
        with self.session_factory() as session:
            return (
                session.scalar(
                    select(func.count()).select_from(
                        DocumentSnapshotRow
                    )
                ),
                session.scalar(
                    select(func.count()).select_from(EvidenceChunkRow)
                ),
                session.scalar(
                    select(func.count()).select_from(
                        DocumentSnapshotChunkRow
                    )
                ),
            )

    @staticmethod
    def _register_artifact(
        manifest_path: Path,
        artifact_root: Path,
        artifact_path: Path,
        *,
        captured_at: datetime,
        replace_existing: bool = False,
    ) -> None:
        register_corpus_artifact(
            manifest_path,
            artifact_root=artifact_root,
            artifact_path=artifact_path,
            source_id="example-2027-backend",
            company="Example Technology",
            job_title="Backend Engineer",
            job_id="backend-01",
            recruitment_type="campus",
            published_date=None,
            location="Shanghai",
            source_kind="official_company",
            source_url="https://careers.example/jobs/backend",
            final_url="https://careers.example/jobs/backend.html",
            captured_at=captured_at,
            status="active",
            replace_existing=replace_existing,
        )


if __name__ == "__main__":
    unittest.main()
