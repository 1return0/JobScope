import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from app.application.documents.document_parsing import (
    DocumentParserRegistry,
    DocumentParsingService,
)
from app.domain.documents.document_ingestion import inspect_document_artifact
from app.infrastructure.documents.html_document_parser import HtmlDocumentParser


class DocumentParsingFailureClassificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DocumentParsingService(
            DocumentParserRegistry([HtmlDocumentParser()])
        )

    def test_success_contains_document_and_no_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact = self._write_artifact(
                temporary_directory,
                "notice.html",
                b"<h1>Campus Recruitment</h1>",
            )

            result = self.service.parse(artifact)

        self.assertTrue(result.succeeded)
        self.assertIsNotNone(result.parsed_document)
        self.assertIsNone(result.failure)

    def test_missing_parser_has_stable_failure_code(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact = self._write_artifact(
                temporary_directory,
                "notice.pdf",
                b"%PDF-placeholder",
            )

            result = self.service.parse(artifact)

        self.assertFalse(result.succeeded)
        self.assertEqual("parser-not-registered", result.failure.code)
        self.assertFalse(result.failure.retryable)
        self.assertEqual(
            "configure-parser",
            result.failure.operator_action,
        )

    def test_format_mismatch_is_classified_separately(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            html_artifact = self._write_artifact(
                temporary_directory,
                "notice.html",
                b"<p>Content</p>",
            )
            routed_as_pdf = replace(
                html_artifact,
                document_format="pdf",
            )
            service = DocumentParsingService(
                DocumentParserRegistry(
                    [
                        HtmlDocumentParser(),
                        _MismatchedHtmlParser(),
                    ]
                )
            )

            result = service.parse(routed_as_pdf)

        self.assertEqual("format-mismatch", result.failure.code)
        self.assertEqual(
            "investigate-format-routing",
            result.failure.operator_action,
        )

    def test_missing_local_file_is_retryable_read_failure(self) -> None:
        temporary_directory = TemporaryDirectory()
        try:
            artifact = self._write_artifact(
                temporary_directory.name,
                "notice.html",
                b"<p>Content</p>",
            )
            artifact.path.unlink()

            result = self.service.parse(artifact)
        finally:
            temporary_directory.cleanup()

        self.assertEqual("document-read-failed", result.failure.code)
        self.assertTrue(result.failure.retryable)
        self.assertEqual(
            "retry-source-read",
            result.failure.operator_action,
        )

    def test_non_utf8_html_requires_encoding_fallback(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact = self._write_artifact(
                temporary_directory,
                "notice.html",
                b"<p>\xff\xfe</p>",
            )

            result = self.service.parse(artifact)

        self.assertEqual(
            "unsupported-text-encoding",
            result.failure.code,
        )
        self.assertFalse(result.failure.retryable)
        self.assertEqual(
            "detect-or-transcode-encoding",
            result.failure.operator_action,
        )

    def test_empty_html_is_routed_to_fallback_or_review(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact = self._write_artifact(
                temporary_directory,
                "notice.html",
                b"<html><body><div></div></body></html>",
            )

            result = self.service.parse(artifact)

        self.assertEqual("no-extractable-content", result.failure.code)
        self.assertEqual(
            "use-fallback-extractor-or-manual-review",
            result.failure.operator_action,
        )

    def test_unexpected_programming_error_is_not_hidden_as_file_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact = self._write_artifact(
                temporary_directory,
                "notice.html",
                b"<p>Content</p>",
            )
            service = DocumentParsingService(
                DocumentParserRegistry([_BrokenParser()])
            )

            with self.assertRaisesRegex(RuntimeError, "parser bug"):
                service.parse(artifact)

    @staticmethod
    def _write_artifact(
        temporary_directory: str,
        filename: str,
        content: bytes,
    ):
        path = Path(temporary_directory) / filename
        path.write_bytes(content)
        return inspect_document_artifact(path, f"source-{filename}")


class _MismatchedHtmlParser:
    document_format = "pdf"

    def parse(self, artifact):
        return HtmlDocumentParser().parse(artifact)


class _BrokenParser:
    document_format = "html"

    def parse(self, artifact):
        del artifact
        raise RuntimeError("parser bug")


if __name__ == "__main__":
    unittest.main()
