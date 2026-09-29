import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from app.domain.documents.document_chunking import (
    EvidenceChunk,
    StructureAwareChunker,
)
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    EvidenceLocation,
    ParsedDocument,
    ParsedFragment,
)
from app.infrastructure.persistence.database import (
    Base,
    create_session_factory,
)
from app.infrastructure.persistence.document_corpus_repository import (
    InvalidCorpusChunksError,
    SqlAlchemyDocumentCorpusRepository,
)
from app.infrastructure.persistence.evidence_chunk_repository import (
    SqlAlchemyEvidenceChunkRepository,
)
from app.infrastructure.persistence.models import (
    DocumentSnapshotChunkRow,
    DocumentSnapshotRow,
    EvidenceChunkRow,
)


class SqlAlchemyDocumentCorpusRepositoryTest(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_persists_snapshot_chunks_and_links_atomically(
        self,
    ) -> None:
        artifact, chunks = self._document("a")

        report = self.repository.persist(
            artifact,
            chunks,
            observed_at=self._time(9),
        )

        self.assertTrue(report.snapshot.created)
        self.assertEqual(len(chunks), report.chunks.inserted)
        self.assertEqual(len(chunks), report.links.inserted)
        self.assertEqual(
            (1, len(chunks), len(chunks)),
            self._row_counts(),
        )

    def test_repeated_persistence_is_idempotent(self) -> None:
        artifact, chunks = self._document("a")
        self.repository.persist(
            artifact,
            chunks,
            observed_at=self._time(9),
        )

        repeated = self.repository.persist(
            artifact,
            chunks,
            observed_at=self._time(11),
        )

        self.assertFalse(repeated.snapshot.created)
        self.assertEqual(0, repeated.chunks.inserted)
        self.assertEqual(0, repeated.links.inserted)
        self.assertEqual(
            (1, len(chunks), len(chunks)),
            self._row_counts(),
        )

    def test_late_chunk_failure_rolls_back_snapshot(self) -> None:
        artifact, chunks = self._document("a")
        valid = chunks[0]
        invalid_row = EvidenceChunkRow(
            evidence_id=valid.evidence_id,
            document_sha256=valid.document_sha256,
            source_reference=valid.source_reference,
            source_fragment_ordinal=-1,
            chunk_ordinal=valid.chunk_ordinal,
            chunker_version=valid.chunker_version,
            text=valid.text,
            page_number=valid.location.page_number,
            heading_path=list(valid.location.heading_path),
        )

        with patch.object(
            SqlAlchemyEvidenceChunkRepository,
            "_to_row",
            return_value=invalid_row,
        ):
            with self.assertRaises(IntegrityError):
                self.repository.persist(
                    artifact,
                    [valid],
                    observed_at=self._time(9),
                )

        self.assertEqual((0, 0, 0), self._row_counts())

    def test_rejects_chunk_from_another_artifact_before_writing(
        self,
    ) -> None:
        artifact, _ = self._document("a")
        _, foreign_chunks = self._document("b")

        with self.assertRaisesRegex(
            InvalidCorpusChunksError,
            "must belong",
        ):
            self.repository.persist(
                artifact,
                foreign_chunks,
                observed_at=self._time(9),
            )

        self.assertEqual((0, 0, 0), self._row_counts())

    def _row_counts(self) -> tuple[int, int, int]:
        with self.session_factory() as session:
            snapshots = session.scalar(
                select(func.count()).select_from(
                    DocumentSnapshotRow
                )
            )
            chunks = session.scalar(
                select(func.count()).select_from(EvidenceChunkRow)
            )
            links = session.scalar(
                select(func.count()).select_from(
                    DocumentSnapshotChunkRow
                )
            )
        return snapshots, chunks, links

    @staticmethod
    def _document(
        hash_character: str,
    ) -> tuple[DocumentArtifact, list[EvidenceChunk]]:
        artifact = DocumentArtifact(
            path=Path("temporary/notice.pdf"),
            source_reference="source-001",
            document_format="pdf",
            content_sha256=hash_character * 64,
            byte_size=1024,
        )
        parsed = ParsedDocument(
            artifact=artifact,
            fragments=(
                ParsedFragment(
                    ordinal=0,
                    text="AI Agent Intern",
                    location=EvidenceLocation(page_number=3),
                ),
            ),
        )
        return artifact, list(
            StructureAwareChunker().chunk(parsed)
        )

    @staticmethod
    def _time(hour: int) -> datetime:
        return datetime(2026, 7, 28, hour, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
