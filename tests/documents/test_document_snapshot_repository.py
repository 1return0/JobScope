import unittest
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from app.domain.documents.document_ingestion import DocumentArtifact
from app.infrastructure.persistence.database import (
    Base,
    create_session_factory,
)
from app.infrastructure.persistence.document_snapshot_repository import (
    SqlAlchemyDocumentSnapshotRepository,
)
from app.infrastructure.persistence.models import DocumentSnapshotRow


class SqlAlchemyDocumentSnapshotRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.repository = SqlAlchemyDocumentSnapshotRepository(
            self.session_factory
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_first_observation_creates_current_version_one(
        self,
    ) -> None:
        report = self.repository.record(
            self._artifact("a"),
            observed_at=self._time(9),
        )

        self.assertTrue(report.created)
        self.assertFalse(report.content_changed)
        self.assertEqual(1, report.snapshot.version_number)
        self.assertTrue(report.snapshot.is_current)
        self.assertEqual(
            report.snapshot,
            self.repository.get_current("source-001"),
        )

    def test_same_content_updates_last_observed_without_new_version(
        self,
    ) -> None:
        first = self.repository.record(
            self._artifact("a"),
            observed_at=self._time(9),
        )
        repeated = self.repository.record(
            self._artifact("a"),
            observed_at=self._time(11),
        )

        history = self.repository.list_history("source-001")
        self.assertFalse(repeated.created)
        self.assertFalse(repeated.content_changed)
        self.assertEqual(
            first.snapshot.snapshot_id,
            repeated.snapshot.snapshot_id,
        )
        self.assertEqual(1, len(history))
        self.assertEqual(self._time(11), history[0].last_observed_at)

    def test_changed_content_creates_next_version_and_supersedes_old(
        self,
    ) -> None:
        first = self.repository.record(
            self._artifact("a"),
            observed_at=self._time(9),
        )
        changed = self.repository.record(
            self._artifact("b"),
            observed_at=self._time(11),
        )

        history = self.repository.list_history("source-001")
        current = self.repository.get_current("source-001")
        self.assertTrue(changed.created)
        self.assertTrue(changed.content_changed)
        self.assertEqual(2, changed.snapshot.version_number)
        self.assertFalse(history[0].is_current)
        self.assertTrue(history[1].is_current)
        self.assertNotEqual(
            first.snapshot.snapshot_id,
            changed.snapshot.snapshot_id,
        )
        self.assertEqual(changed.snapshot, current)

    def test_rejects_observation_without_timezone(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.repository.record(
                self._artifact("a"),
                observed_at=datetime(2026, 7, 28, 9),
            )

    def test_database_rejects_two_current_versions_for_one_source(
        self,
    ) -> None:
        self.repository.record(
            self._artifact("a"),
            observed_at=self._time(9),
        )
        self.repository.record(
            self._artifact("b"),
            observed_at=self._time(11),
        )

        with self.assertRaises(IntegrityError):
            with self.session_factory.begin() as session:
                old_row = session.scalar(
                    select(DocumentSnapshotRow).where(
                        DocumentSnapshotRow.content_sha256
                        == "a" * 64
                    )
                )
                assert old_row is not None
                old_row.is_current = True
                session.flush()

    @staticmethod
    def _artifact(hash_character: str) -> DocumentArtifact:
        return DocumentArtifact(
            path=Path("temporary/notice.pdf"),
            source_reference="source-001",
            document_format="pdf",
            content_sha256=hash_character * 64,
            byte_size=1024,
        )

    @staticmethod
    def _time(hour: int) -> datetime:
        return datetime(2026, 7, 28, hour, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
