from __future__ import annotations

from pathlib import Path

from app.application.ocr.pdf_ocr_fallback import OcrDocumentAssembler
from app.domain.documents.document_ingestion import inspect_document_artifact
from app.domain.documents.document_ocr import OcrEngine, PdfPageRenderer


class PdfOcrTextRecognizer:
    def __init__(
        self,
        page_renderer: PdfPageRenderer,
        ocr_engine: OcrEngine,
        document_assembler: OcrDocumentAssembler,
        *,
        configuration_identity: str,
    ) -> None:
        if not configuration_identity.strip():
            raise ValueError(
                "OCR recognizer configuration identity must not be blank"
            )
        self._page_renderer = page_renderer
        self._ocr_engine = ocr_engine
        self._document_assembler = document_assembler
        self._identity = (
            f"{ocr_engine.identity};{configuration_identity.strip()}"
        )

    @property
    def identity(self) -> str:
        return self._identity

    def recognize_text(self, artifact_path: Path) -> str:
        artifact = inspect_document_artifact(
            artifact_path,
            source_reference=f"ocr-evaluation:{artifact_path.name}",
        )
        rendered_pages = self._page_renderer.render(artifact.path)
        ocr_pages = tuple(
            self._ocr_engine.recognize(page)
            for page in rendered_pages
        )
        parsed_document = self._document_assembler.assemble(
            artifact,
            ocr_pages,
        )
        return "\n".join(
            fragment.text for fragment in parsed_document.fragments
        )
