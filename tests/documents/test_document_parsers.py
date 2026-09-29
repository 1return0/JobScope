import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from app.application.documents.document_parsing import (
    DocumentParserRegistry,
    DuplicateParserRegistrationError,
    InvalidParserOutputError,
    ParserNotRegisteredError,
)
from app.domain.documents.document_ingestion import (
    DocumentParsingError,
    ParsedDocument,
    ParsedFragment,
    EvidenceLocation,
    inspect_document_artifact,
)
from app.infrastructure.documents.html_document_parser import HtmlDocumentParser


class DocumentParserRegistryTest(unittest.TestCase):
    def test_dispatches_artifact_to_registered_parser(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact = self._write_artifact(
                temporary_directory,
                "notice.html",
                "<h1>Campus Recruitment</h1><p>Apply by Friday.</p>",
            )
            registry = DocumentParserRegistry([HtmlDocumentParser()])

            parsed = registry.parse(artifact)

        self.assertEqual(
            ["Campus Recruitment", "Apply by Friday."],
            [fragment.text for fragment in parsed.fragments],
        )

    def test_rejects_format_without_registered_parser(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.pdf"
            path.write_bytes(b"%PDF-placeholder")
            artifact = inspect_document_artifact(path, "source-pdf")

            with self.assertRaisesRegex(
                ParserNotRegisteredError,
                "pdf",
            ):
                DocumentParserRegistry(
                    [HtmlDocumentParser()]
                ).parse(artifact)

    def test_rejects_duplicate_parser_for_same_format(self) -> None:
        registry = DocumentParserRegistry([HtmlDocumentParser()])

        with self.assertRaisesRegex(
            DuplicateParserRegistrationError,
            "html",
        ):
            registry.register(HtmlDocumentParser())

    def test_rejects_parser_output_for_another_artifact(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            first = self._write_artifact(
                temporary_directory,
                "first.html",
                "<p>First</p>",
            )
            second = self._write_artifact(
                temporary_directory,
                "second.html",
                "<p>Second</p>",
            )

            class IncorrectParser:
                document_format = "html"

                def parse(self, artifact):
                    del artifact
                    return ParsedDocument(
                        artifact=second,
                        fragments=(
                            ParsedFragment(
                                ordinal=0,
                                text="wrong artifact",
                                location=EvidenceLocation(),
                            ),
                        ),
                    )

            with self.assertRaises(InvalidParserOutputError):
                DocumentParserRegistry(
                    [IncorrectParser()]
                ).parse(first)

    @staticmethod
    def _write_artifact(
        temporary_directory: str,
        filename: str,
        html: str,
    ):
        path = Path(temporary_directory) / filename
        path.write_text(html, encoding="utf-8")
        return inspect_document_artifact(path, f"source-{filename}")


class HtmlDocumentParserTest(unittest.TestCase):
    def test_preserves_heading_hierarchy_as_evidence_location(self) -> None:
        html = """
        <h1>2027 Campus Recruitment</h1>
        <p>Open to graduating students.</p>
        <h2>Backend Engineer</h2>
        <p>Major is not restricted.</p>
        <h2>Product Manager</h2>
        <ul><li>Submit before October.</li></ul>
        """
        with TemporaryDirectory() as temporary_directory:
            artifact = DocumentParserRegistryTest._write_artifact(
                temporary_directory,
                "notice.html",
                html,
            )

            parsed = HtmlDocumentParser().parse(artifact)

        self.assertEqual(
            ("2027 Campus Recruitment",),
            parsed.fragments[1].location.heading_path,
        )
        self.assertEqual(
            ("2027 Campus Recruitment", "Backend Engineer"),
            parsed.fragments[3].location.heading_path,
        )
        self.assertEqual(
            ("2027 Campus Recruitment", "Product Manager"),
            parsed.fragments[5].location.heading_path,
        )

    def test_ignores_script_and_style_content(self) -> None:
        html = """
        <style>body { display: none; }</style>
        <script>window.fakeDeadline = "yesterday";</script>
        <p>Real recruitment content.</p>
        """
        with TemporaryDirectory() as temporary_directory:
            artifact = DocumentParserRegistryTest._write_artifact(
                temporary_directory,
                "notice.html",
                html,
            )

            parsed = HtmlDocumentParser().parse(artifact)

        self.assertEqual(
            ["Real recruitment content."],
            [fragment.text for fragment in parsed.fragments],
        )

    def test_rejects_html_without_supported_text_blocks(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            artifact = DocumentParserRegistryTest._write_artifact(
                temporary_directory,
                "empty.html",
                "<html><body><div></div></body></html>",
            )

            with self.assertRaisesRegex(
                DocumentParsingError,
                "no supported textual content",
            ):
                HtmlDocumentParser().parse(artifact)

    def test_rejects_non_html_artifact_when_called_directly(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            html_artifact = DocumentParserRegistryTest._write_artifact(
                temporary_directory,
                "notice.html",
                "<p>Content</p>",
            )
            pdf_artifact = replace(
                html_artifact,
                document_format="pdf",
            )

            with self.assertRaisesRegex(
                DocumentParsingError,
                "only accepts html",
            ):
                HtmlDocumentParser().parse(pdf_artifact)


if __name__ == "__main__":
    unittest.main()
