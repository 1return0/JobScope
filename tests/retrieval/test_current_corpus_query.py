import unittest
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.domain.documents.document_chunking import (
    ChunkingConfig,
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
    SqlAlchemyDocumentCorpusRepository,
)


class CurrentCorpusQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self.repository = SqlAlchemyDocumentCorpusRepository(
            create_session_factory(self.engine)
        )
        Base.metadata.create_all(self.engine)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_excludes_chunks_from_superseded_snapshot(self) -> None:
        old_artifact, old_chunks = self._document(
            source_reference="source-001",
            hash_character="a",
            text="Python 3.10",
        )
        new_artifact, new_chunks = self._document(
            source_reference="source-001",
            hash_character="b",
            text="Python 3.12",
        )
        self._persist(old_artifact, old_chunks, hour=9)
        self._persist(new_artifact, new_chunks, hour=11)

        current = self.repository.list_current_chunks(
            chunker_version=new_chunks[0].chunker_version
        )

        self.assertEqual(["Python 3.12"], self._texts(current))

    def test_can_isolate_one_source(self) -> None:
        first_artifact, first_chunks = self._document(
            source_reference="source-001",
            hash_character="a",
            text="AI Agent Intern",
        )
        second_artifact, second_chunks = self._document(
            source_reference="source-002",
            hash_character="b",
            text="Java Backend Intern",
        )
        self._persist(first_artifact, first_chunks, hour=9)
        self._persist(second_artifact, second_chunks, hour=10)

        current = self.repository.list_current_chunks(
            chunker_version=first_chunks[0].chunker_version,
            source_reference="source-002",
        )

        self.assertEqual(
            ["Java Backend Intern"],
            self._texts(current),
        )

    def test_does_not_mix_chunker_versions(self) -> None:
        artifact, first_chunks = self._document(
            source_reference="source-001",
            hash_character="a",
            text="AI Agent Intern requires Python and RAG",
            config=ChunkingConfig(
                max_chars=20,
                overlap_chars=2,
                algorithm_version="structure-v1",
            ),
        )
        _, second_chunks = self._document(
            source_reference="source-001",
            hash_character="a",
            text="AI Agent Intern requires Python and RAG",
            config=ChunkingConfig(
                max_chars=12,
                overlap_chars=2,
                algorithm_version="structure-v2",
            ),
        )
        self._persist(artifact, first_chunks, hour=9)
        self._persist(artifact, second_chunks, hour=10)

        current = self.repository.list_current_chunks(
            chunker_version=first_chunks[0].chunker_version
        )

        self.assertEqual(
            [chunk.evidence_id for chunk in first_chunks],
            [chunk.evidence_id for chunk in current],
        )
        self.assertTrue(
            all(
                chunk.chunker_version
                == first_chunks[0].chunker_version
                for chunk in current
            )
        )

    def _persist(
        self,
        artifact: DocumentArtifact,
        chunks: list[EvidenceChunk],
        *,
        hour: int,
    ) -> None:
        self.repository.persist(
            artifact,
            chunks,
            observed_at=datetime(
                2026,
                7,
                28,
                hour,
                tzinfo=UTC,
            ),
        )

    @staticmethod
    def _document(
        *,
        source_reference: str,
        hash_character: str,
        text: str,
        config: ChunkingConfig = ChunkingConfig(),
    ) -> tuple[DocumentArtifact, list[EvidenceChunk]]:
        artifact = DocumentArtifact(
            path=Path("temporary/notice.pdf"),
            source_reference=source_reference,
            document_format="pdf",
            content_sha256=hash_character * 64,
            byte_size=1024,
        )
        parsed = ParsedDocument(
            artifact=artifact,
            fragments=(
                ParsedFragment(
                    ordinal=0,
                    text=text,
                    location=EvidenceLocation(page_number=3),
                ),
            ),
        )
        return artifact, list(
            StructureAwareChunker(config).chunk(parsed)
        )

    @staticmethod
    def _texts(chunks: list[EvidenceChunk]) -> list[str]:
        return [chunk.text for chunk in chunks]


if __name__ == "__main__":
    unittest.main()
