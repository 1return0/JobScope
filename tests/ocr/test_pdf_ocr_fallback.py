import unittest
from pathlib import Path

from app.application.ocr.pdf_ocr_fallback import PdfOcrFallbackParser
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    DocumentReadError,
    EvidenceLocation,
    ParsedDocument,
    ParsedFragment,
    PdfTextLayerMissingError,
)
from app.domain.documents.document_ocr import (
    OcrPageResult,
    RenderedPdfPage,
)


def _artifact() -> DocumentArtifact:
    return DocumentArtifact(
        path=Path("scan.pdf"),
        source_reference="https://example.edu/scan.pdf",
        document_format="pdf",
        content_sha256="a" * 64,
        byte_size=100,
    )


def _parsed_document(artifact: DocumentArtifact) -> ParsedDocument:
    return ParsedDocument(
        artifact=artifact,
        fragments=(
            ParsedFragment(
                ordinal=0,
                text="text layer",
                location=EvidenceLocation(page_number=1),
            ),
        ),
    )


class _PrimaryParser:
    document_format = "pdf"

    def __init__(self, result: ParsedDocument | Exception) -> None:
        self._result = result

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _Renderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, path: Path) -> tuple[RenderedPdfPage, ...]:
        self.calls += 1
        return (
            RenderedPdfPage(1, 100, 200, 300, b"page-1"),
            RenderedPdfPage(2, 100, 200, 300, b"page-2"),
        )


class _OcrEngine:
    identity = "test-ocr;revision=1"

    def __init__(self) -> None:
        self.page_numbers: list[int] = []

    def recognize(self, page: RenderedPdfPage) -> OcrPageResult:
        self.page_numbers.append(page.page_number)
        return OcrPageResult(
            page_number=page.page_number,
            engine_identity=self.identity,
            regions=(),
        )


class _Assembler:
    def __init__(self, result: ParsedDocument) -> None:
        self._result = result
        self.received_pages: tuple[OcrPageResult, ...] | None = None

    def assemble(
        self,
        artifact: DocumentArtifact,
        pages: tuple[OcrPageResult, ...],
    ) -> ParsedDocument:
        self.received_pages = pages
        return self._result


class PdfOcrFallbackParserTest(unittest.TestCase):
    def test_returns_text_layer_result_without_running_ocr(self) -> None:
        artifact = _artifact()
        expected = _parsed_document(artifact)
        renderer = _Renderer()
        engine = _OcrEngine()
        parser = PdfOcrFallbackParser(
            _PrimaryParser(expected),
            renderer,
            engine,
            _Assembler(expected),
        )

        actual = parser.parse(artifact)

        self.assertIs(expected, actual)
        self.assertEqual(0, renderer.calls)
        self.assertEqual([], engine.page_numbers)

    def test_runs_ocr_for_each_page_only_when_text_layer_is_missing(self) -> None:
        artifact = _artifact()
        expected = _parsed_document(artifact)
        renderer = _Renderer()
        engine = _OcrEngine()
        assembler = _Assembler(expected)
        parser = PdfOcrFallbackParser(
            _PrimaryParser(PdfTextLayerMissingError("no text layer")),
            renderer,
            engine,
            assembler,
        )

        actual = parser.parse(artifact)

        self.assertIs(expected, actual)
        self.assertEqual(1, renderer.calls)
        self.assertEqual([1, 2], engine.page_numbers)
        self.assertEqual(
            (1, 2),
            tuple(page.page_number for page in assembler.received_pages or ()),
        )

    def test_does_not_hide_an_unreadable_pdf_as_an_ocr_fallback(self) -> None:
        artifact = _artifact()
        renderer = _Renderer()
        expected = _parsed_document(artifact)
        parser = PdfOcrFallbackParser(
            _PrimaryParser(DocumentReadError("read failed")),
            renderer,
            _OcrEngine(),
            _Assembler(expected),
        )

        with self.assertRaises(DocumentReadError):
            parser.parse(artifact)

        self.assertEqual(0, renderer.calls)


if __name__ == "__main__":
    unittest.main()
