import unittest
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
from app.infrastructure.persistence.evidence_chunk_repository import (
    SqlAlchemyEvidenceChunkRepository,
)


class SqlAlchemyEvidenceChunkRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.repository = SqlAlchemyEvidenceChunkRepository(
            create_session_factory(self.engine)
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_round_trips_text_and_evidence_location(self) -> None:
        chunk = self._chunks()[0]

        report = self.repository.save_many([chunk])
        loaded = self.repository.list_by_document(
            chunk.document_sha256
        )

        self.assertEqual(1, report.inserted)
        self.assertEqual([chunk], loaded)
        self.assertEqual(
            ("Campus Recruitment", "AI Agent Intern"),
            loaded[0].location.heading_path,
        )
        self.assertEqual(3, loaded[0].location.page_number)

    def test_repeated_save_is_idempotent(self) -> None:
        chunks = self._chunks()

        first = self.repository.save_many(
            [*chunks, chunks[0]]
        )
        second = self.repository.save_many(chunks)
        loaded = self.repository.list_by_document(
            chunks[0].document_sha256
        )

        self.assertEqual(len(chunks) + 1, first.total)
        self.assertEqual(len(chunks), first.inserted)
        self.assertEqual(1, first.skipped)
        self.assertEqual(0, second.inserted)
        self.assertEqual(len(chunks), second.skipped)
        self.assertEqual(len(chunks), len(loaded))

    def test_query_isolates_document_hash(self) -> None:
        first = self._chunks(document_sha256="a" * 64)[0]
        second = self._chunks(document_sha256="b" * 64)[0]
        self.repository.save_many([first, second])

        loaded = self.repository.list_by_document("a" * 64)

        self.assertEqual([first], loaded)

    def test_query_can_filter_chunker_version(self) -> None:
        first = self._chunks(
            config=ChunkingConfig(
                max_chars=100,
                overlap_chars=10,
                algorithm_version="structure-v1",
            )
        )[0]
        second = self._chunks(
            config=ChunkingConfig(
                max_chars=100,
                overlap_chars=10,
                algorithm_version="structure-v2",
            )
        )[0]
        self.repository.save_many([first, second])

        loaded = self.repository.list_by_document(
            first.document_sha256,
            chunker_version=first.chunker_version,
        )

        self.assertEqual([first], loaded)

    @staticmethod
    def _chunks(
        *,
        document_sha256: str = "a" * 64,
        config: ChunkingConfig = ChunkingConfig(
            max_chars=18,
            overlap_chars=3,
        ),
    ) -> list[EvidenceChunk]:
        document = ParsedDocument(
            artifact=DocumentArtifact(
                path=Path("temporary/notice.pdf"),
                source_reference=(
                    "https://careers.example/notice.pdf"
                ),
                document_format="pdf",
                content_sha256=document_sha256,
                byte_size=2048,
            ),
            fragments=(
                ParsedFragment(
                    ordinal=0,
                    text="AI Agent Intern requires Python and RAG skills",
                    location=EvidenceLocation(
                        page_number=3,
                        heading_path=(
                            "Campus Recruitment",
                            "AI Agent Intern",
                        ),
                    ),
                ),
            ),
        )
        return list(StructureAwareChunker(config).chunk(document))


if __name__ == "__main__":
    unittest.main()
