import unittest
from pathlib import Path

from app.application.documents.document_parsing import (
    DocumentParserRegistry,
    DocumentParsingService,
)
from app.application.documents.document_processing import (
    DocumentProcessingResult,
    DocumentProcessingService,
)
from app.domain.documents.document_chunking import (
    ChunkingConfig,
    StructureAwareChunker,
)
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    EvidenceLocation,
    NoExtractableContentError,
    ParsedDocument,
    ParsedFragment,
)


class _SuccessfulParser:
    document_format = "html"

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        return ParsedDocument(
            artifact=artifact,
            fragments=(
                ParsedFragment(
                    ordinal=0,
                    text="A" * 30,
                    location=EvidenceLocation(
                        heading_path=("AI Agent Intern",)
                    ),
                ),
            ),
        )


class _FailedParser:
    document_format = "html"

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        raise NoExtractableContentError("document is empty")


class DocumentProcessingServiceTest(unittest.TestCase):
    def test_successful_parse_is_chunked_with_preserved_location(
        self,
    ) -> None:
        service = self._service(_SuccessfulParser())

        result = service.process(self._artifact())

        self.assertTrue(result.succeeded)
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(
            all(
                chunk.location.heading_path == ("AI Agent Intern",)
                for chunk in result.chunks
            )
        )

    def test_known_parsing_failure_stops_before_chunking(self) -> None:
        service = self._service(_FailedParser())

        result = service.process(self._artifact())

        self.assertFalse(result.succeeded)
        self.assertEqual((), result.chunks)
        self.assertEqual(
            "no-extractable-content",
            result.parsing_result.failure.code,
        )

    def test_failed_result_rejects_fabricated_chunks(self) -> None:
        failed = self._service(_FailedParser()).process(
            self._artifact()
        )
        successful = self._service(_SuccessfulParser()).process(
            self._artifact()
        )

        with self.assertRaisesRegex(ValueError, "must not contain"):
            DocumentProcessingResult(
                parsing_result=failed.parsing_result,
                chunks=successful.chunks,
            )

    @staticmethod
    def _service(parser) -> DocumentProcessingService:
        parsing_service = DocumentParsingService(
            DocumentParserRegistry([parser])
        )
        return DocumentProcessingService(
            parsing_service,
            StructureAwareChunker(
                ChunkingConfig(max_chars=12, overlap_chars=2)
            ),
        )

    @staticmethod
    def _artifact() -> DocumentArtifact:
        return DocumentArtifact(
            path=Path("temporary/notice.html"),
            source_reference="https://careers.example/notice.html",
            document_format="html",
            content_sha256="c" * 64,
            byte_size=256,
        )


if __name__ == "__main__":
    unittest.main()
