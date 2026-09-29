from __future__ import annotations

from typing import Protocol

from app.application.documents.document_parsing import DocumentParser
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    DocumentFormat,
    ParsedDocument,
    PdfTextLayerMissingError,
)
from app.domain.documents.document_ocr import (
    OcrEngine,
    OcrPageResult,
    PdfPageRenderer,
)


class OcrDocumentAssembler(Protocol):
    def assemble(
        self,
        artifact: DocumentArtifact,
        pages: tuple[OcrPageResult, ...],
    ) -> ParsedDocument:
        ...


class PdfOcrFallbackParser:
    document_format: DocumentFormat = "pdf"

    def __init__(
        self,
        primary_parser: DocumentParser,
        page_renderer: PdfPageRenderer,
        ocr_engine: OcrEngine,
        document_assembler: OcrDocumentAssembler,
    ) -> None:
        if primary_parser.document_format != self.document_format:
            raise ValueError("primary parser must handle PDF documents")
        self._primary_parser = primary_parser
        self._page_renderer = page_renderer
        self._ocr_engine = ocr_engine
        self._document_assembler = document_assembler

    def parse(self, artifact: DocumentArtifact) -> ParsedDocument:
        try:
            return self._primary_parser.parse(artifact)
        except PdfTextLayerMissingError:
            rendered_pages = self._page_renderer.render(artifact.path)
            ocr_pages = tuple(
                self._ocr_engine.recognize(page)
                for page in rendered_pages
            )
            return self._document_assembler.assemble(
                artifact,
                ocr_pages,
            )
