import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from app.bootstrap import (
    _build_pdf_parser,
    build_document_parser_registry,
    build_document_parsing_service,
    build_document_processing_service,
)
from app.application.documents.document_parsing import DocumentParsingService
from app.config import Settings
from app.domain.documents.document_ingestion import inspect_document_artifact
from app.infrastructure.documents.pdf_document_parser import PdfDocumentParser


class DocumentParserBootstrapTest(unittest.TestCase):
    def test_processing_service_reuses_an_existing_parsing_service(
        self,
    ) -> None:
        parsing_service = Mock(
            spec=DocumentParsingService
        )

        with patch(
            "app.bootstrap.DocumentProcessingService"
        ) as processing_service_type:
            build_document_processing_service(
                Settings(),
                parsing_service=parsing_service,
            )

        actual_parsing_service = (
            processing_service_type.call_args.args[0]
        )
        self.assertIs(parsing_service, actual_parsing_service)

    def test_disabled_ocr_keeps_the_lightweight_pdf_parser(self) -> None:
        with patch(
            "app.bootstrap.build_paddle_ocr_pipeline_predictor"
        ) as build_predictor:
            parser = _build_pdf_parser(
                Settings(pdf_ocr_enabled=False)
            )

        self.assertIsInstance(parser, PdfDocumentParser)
        build_predictor.assert_not_called()

    def test_enabled_ocr_builds_the_fallback_dependencies(self) -> None:
        predictor = object()
        renderer = object()
        settings = Settings(
            pdf_ocr_enabled=True,
            pdf_ocr_engine="test-engine",
            pdf_ocr_minimum_confidence=0.75,
            pdf_ocr_render_dpi=240,
            pdf_ocr_max_pages=25,
            pdf_ocr_max_pixels_per_page=12_000_000,
        )

        with (
            patch(
                "app.bootstrap.build_paddle_ocr_pipeline_predictor",
                return_value=predictor,
            ) as build_predictor,
            patch(
                "app.bootstrap.build_pdfium_page_renderer",
                return_value=renderer,
            ) as build_renderer,
            patch(
                "app.bootstrap.PaddleOcrTextEngine"
            ) as engine_type,
            patch(
                "app.bootstrap.PdfOcrFallbackParser",
                return_value=object(),
            ) as fallback_type,
        ):
            _build_pdf_parser(settings)

        build_predictor.assert_called_once_with(
            engine="test-engine",
            cache_folder=settings.pdf_ocr_cache_folder,
        )
        renderer_config = build_renderer.call_args.args[0]
        self.assertEqual(240, renderer_config.dpi)
        self.assertEqual(25, renderer_config.max_pages)
        self.assertEqual(
            12_000_000,
            renderer_config.max_pixels_per_page,
        )
        engine_type.assert_called_once_with(
            predictor,
            minimum_confidence=0.75,
        )
        fallback_type.assert_called_once()

    def test_default_registry_contains_every_supported_format(self) -> None:
        registry = build_document_parser_registry()

        self.assertSetEqual(
            {"html", "pdf", "docx", "xlsx", "pptx"},
            set(registry.registered_formats),
        )

    def test_default_parsing_service_uses_registered_parser(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "notice.html"
            path.write_text(
                "<h1>Campus Recruitment</h1>",
                encoding="utf-8",
            )
            artifact = inspect_document_artifact(
                path,
                "source-bootstrap-html",
            )

            result = build_document_parsing_service().parse(artifact)

        self.assertTrue(result.succeeded)
        self.assertEqual(
            "Campus Recruitment",
            result.parsed_document.fragments[0].text,
        )


if __name__ == "__main__":
    unittest.main()
