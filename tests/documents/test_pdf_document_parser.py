import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from app.application.documents.document_parsing import (
    DocumentParserRegistry,
    DocumentParsingService,
)
from app.domain.documents.document_ingestion import (
    DocumentParsingError,
    inspect_document_artifact,
)
from app.infrastructure.documents.pdf_document_parser import PdfDocumentParser


class PdfDocumentParserTest(unittest.TestCase):
    def test_extracts_text_with_original_page_numbers(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "recruitment.pdf"
            writer = PdfWriter()
            self._add_text_page(writer, "2027 Campus Recruitment")
            writer.add_blank_page(width=612, height=792)
            self._add_text_page(writer, "AI Agent Intern")
            self._write_pdf(writer, path)
            artifact = inspect_document_artifact(
                path,
                "https://careers.example/recruitment.pdf",
            )

            parsed = PdfDocumentParser().parse(artifact)

        self.assertEqual(
            ["2027 Campus Recruitment", "AI Agent Intern"],
            [fragment.text for fragment in parsed.fragments],
        )
        self.assertEqual(
            [1, 3],
            [
                fragment.location.page_number
                for fragment in parsed.fragments
            ],
        )
        self.assertEqual([0, 1], [
            fragment.ordinal for fragment in parsed.fragments
        ])

    def test_registry_dispatches_pdf_parser(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.pdf"
            writer = PdfWriter()
            self._add_text_page(writer, "Application deadline")
            self._write_pdf(writer, path)
            artifact = inspect_document_artifact(path, "source-pdf")

            parsed = DocumentParserRegistry(
                [PdfDocumentParser()]
            ).parse(artifact)

        self.assertEqual(
            "Application deadline",
            parsed.fragments[0].text,
        )

    def test_pdf_without_text_layer_is_routed_to_ocr(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scanned.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            self._write_pdf(writer, path)
            artifact = inspect_document_artifact(path, "source-scan")
            service = DocumentParsingService(
                DocumentParserRegistry([PdfDocumentParser()])
            )

            result = service.parse(artifact)

        self.assertEqual(
            "pdf-text-layer-missing",
            result.failure.code,
        )
        self.assertEqual(
            "run-ocr-or-manual-review",
            result.failure.operator_action,
        )

    def test_corrupt_pdf_has_stable_failure_classification(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "corrupt.pdf"
            path.write_bytes(b"not-a-pdf")
            artifact = inspect_document_artifact(path, "source-corrupt")
            service = DocumentParsingService(
                DocumentParserRegistry([PdfDocumentParser()])
            )

            with self.assertLogs("pypdf", level="WARNING"):
                result = service.parse(artifact)

        self.assertEqual("invalid-pdf", result.failure.code)
        self.assertEqual(
            "redownload-or-manual-review",
            result.failure.operator_action,
        )

    def test_password_protected_pdf_is_classified(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "protected.pdf"
            writer = PdfWriter()
            self._add_text_page(writer, "Confidential recruitment")
            writer.encrypt("secret-password")
            self._write_pdf(writer, path)
            artifact = inspect_document_artifact(path, "source-protected")
            service = DocumentParsingService(
                DocumentParserRegistry([PdfDocumentParser()])
            )

            result = service.parse(artifact)

        self.assertEqual(
            "password-protected-document",
            result.failure.code,
        )
        self.assertEqual(
            "request-unlocked-document-or-manual-review",
            result.failure.operator_action,
        )

    def test_rejects_non_pdf_artifact_when_called_directly(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.pdf"
            writer = PdfWriter()
            self._add_text_page(writer, "Content")
            self._write_pdf(writer, path)
            pdf_artifact = inspect_document_artifact(
                path,
                "source-pdf",
            )
            docx_artifact = replace(
                pdf_artifact,
                document_format="docx",
            )

            with self.assertRaisesRegex(
                DocumentParsingError,
                "only accepts pdf",
            ):
                PdfDocumentParser().parse(docx_artifact)

    @staticmethod
    def _add_text_page(writer: PdfWriter, text: str) -> None:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): writer._add_object(font),
                    }
                )
            }
        )
        content = DecodedStreamObject()
        content.set_data(
            (
                "BT /F1 12 Tf 72 720 Td "
                f"({text}) Tj ET"
            ).encode("ascii")
        )
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = writer._add_object(content)

    @staticmethod
    def _write_pdf(writer: PdfWriter, path: Path) -> None:
        with path.open("wb") as pdf_file:
            writer.write(pdf_file)


if __name__ == "__main__":
    unittest.main()
