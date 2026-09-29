import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain.documents.document_ingestion import (
    EvidenceLocation,
    ParsedDocument,
    ParsedFragment,
)
from app.domain.documents.document_ocr import (
    OcrPageResult,
    RenderedPdfPage,
)
from app.infrastructure.ocr.pdf_ocr_text_recognizer import (
    PdfOcrTextRecognizer,
)


class _Renderer:
    def render(self, path: Path) -> tuple[RenderedPdfPage, ...]:
        return (RenderedPdfPage(1, 100, 200, 300, b"png"),)


class _Engine:
    identity = "ocr=test;revision=1"

    def recognize(self, page: RenderedPdfPage) -> OcrPageResult:
        return OcrPageResult(page.page_number, self.identity, ())


class _Assembler:
    def assemble(self, artifact, pages) -> ParsedDocument:
        return ParsedDocument(
            artifact=artifact,
            fragments=(
                ParsedFragment(
                    0,
                    "AI Agent Intern",
                    EvidenceLocation(page_number=1),
                ),
                ParsedFragment(
                    1,
                    "Shanghai",
                    EvidenceLocation(page_number=2),
                ),
            ),
        )


class PdfOcrTextRecognizerTest(unittest.TestCase):
    def test_returns_joined_parsed_text_and_configuration_identity(self) -> None:
        recognizer = PdfOcrTextRecognizer(
            _Renderer(),
            _Engine(),
            _Assembler(),
            configuration_identity="dpi=300;confidence=0.5",
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            path.write_bytes(b"pdf")
            text = recognizer.recognize_text(path)

        self.assertEqual("AI Agent Intern\nShanghai", text)
        self.assertIn("dpi=300", recognizer.identity)
        self.assertIn("ocr=test", recognizer.identity)


if __name__ == "__main__":
    unittest.main()
