import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    EvidenceLocation,
    ParsedDocument,
    ParsedFragment,
    UnsupportedDocumentFormatError,
    detect_document_format,
    inspect_document_artifact,
)


class DocumentIngestionContractTest(unittest.TestCase):
    def test_detects_each_supported_document_format(self) -> None:
        cases = {
            "job.html": "html",
            "job.HTM": "html",
            "notice.PDF": "pdf",
            "notice.docx": "docx",
            "positions.xlsx": "xlsx",
            "presentation.pptx": "pptx",
        }

        for filename, expected_format in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    expected_format,
                    detect_document_format(Path(filename)),
                )

    def test_rejects_unsupported_or_missing_extension(self) -> None:
        for filename in ("resume.doc", "positions.csv", "README"):
            with self.subTest(filename=filename):
                with self.assertRaises(
                    UnsupportedDocumentFormatError
                ):
                    detect_document_format(Path(filename))

    def test_artifact_rejects_unsupported_runtime_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "not supported"):
            DocumentArtifact(
                path=Path("notice.txt"),
                source_reference="source-a",
                document_format="txt",  # type: ignore[arg-type]
                content_sha256="0" * 64,
                byte_size=0,
            )

    def test_inspection_records_content_hash_size_and_source(self) -> None:
        content = b"public recruitment notice"

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.pdf"
            path.write_bytes(content)
            artifact = inspect_document_artifact(
                path,
                source_reference="https://careers.example/notice.pdf",
            )

        self.assertEqual("pdf", artifact.document_format)
        self.assertEqual(len(content), artifact.byte_size)
        self.assertEqual(
            hashlib.sha256(content).hexdigest(),
            artifact.content_sha256,
        )
        self.assertEqual(
            "https://careers.example/notice.pdf",
            artifact.source_reference,
        )

    def test_same_content_has_same_hash_across_file_names(self) -> None:
        content = b"same captured bytes"

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path = root / "first.pdf"
            second_path = root / "second.docx"
            first_path.write_bytes(content)
            second_path.write_bytes(content)

            first = inspect_document_artifact(
                first_path,
                source_reference="source-a",
            )
            second = inspect_document_artifact(
                second_path,
                source_reference="source-b",
            )

        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertNotEqual(first.document_format, second.document_format)

    def test_evidence_location_requires_positive_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "page_number"):
            EvidenceLocation(page_number=0)
        with self.assertRaisesRegex(ValueError, "slide_number"):
            EvidenceLocation(slide_number=-1)
        with self.assertRaisesRegex(ValueError, "sheet_name"):
            EvidenceLocation(cell_reference="A1:B4")
        with self.assertRaisesRegex(ValueError, "table_number"):
            EvidenceLocation(table_number=0)
        with self.assertRaisesRegex(ValueError, "table_number"):
            EvidenceLocation(table_row_number=1)

    def test_parsed_document_uses_one_contract_for_all_formats(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.docx"
            path.write_bytes(b"docx-placeholder")
            artifact = inspect_document_artifact(
                path,
                source_reference="https://example/notice.docx",
            )

            parsed_document = ParsedDocument(
                artifact=artifact,
                fragments=(
                    ParsedFragment(
                        ordinal=0,
                        text="2027 campus recruitment requirements",
                        location=EvidenceLocation(
                            heading_path=("Campus Recruitment",),
                        ),
                    ),
                ),
            )

        self.assertEqual("docx", parsed_document.artifact.document_format)
        self.assertEqual(
            "2027 campus recruitment requirements",
            parsed_document.fragments[0].text,
        )

    def test_parsed_document_requires_contiguous_ordinals(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.html"
            path.write_bytes(b"<p>content</p>")
            artifact = inspect_document_artifact(path, "source-html")

            with self.assertRaisesRegex(
                ValueError,
                "ordinals",
            ):
                ParsedDocument(
                    artifact=artifact,
                    fragments=(
                        ParsedFragment(
                            ordinal=1,
                            text="content",
                            location=EvidenceLocation(),
                        ),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
