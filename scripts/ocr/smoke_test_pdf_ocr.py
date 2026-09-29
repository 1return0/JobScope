from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from app.application.ocr.pdf_ocr_parsing import (
    OcrParsedDocumentAssembler,
    SingleColumnOcrReadingOrder,
)
from app.application.ocr.ocr_evaluation import evaluate_ocr_text
from app.domain.documents.document_ingestion import inspect_document_artifact
from app.infrastructure.ocr.paddle_ocr_engine import PaddleOcrTextEngine
from app.infrastructure.ocr.paddle_ocr_pipeline_predictor import (
    build_paddle_ocr_pipeline_predictor,
)
from app.infrastructure.ocr.pdfium_page_renderer import (
    PdfiumPageRendererConfig,
    build_pdfium_page_renderer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real local PDF render and OCR smoke test without writing "
            "to PostgreSQL."
        )
    )
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--engine", default="transformers")
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=Path(".model-cache") / "paddlex",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--preview-chars", type=int, default=300)
    parser.add_argument("--expected-text-file", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    started_at = perf_counter()
    try:
        artifact = inspect_document_artifact(
            arguments.pdf_path,
            source_reference=f"local-smoke:{arguments.pdf_path.name}",
        )
        renderer = build_pdfium_page_renderer(
            PdfiumPageRendererConfig(
                dpi=arguments.dpi,
                max_pages=arguments.max_pages,
            )
        )
        predictor = build_paddle_ocr_pipeline_predictor(
            engine=arguments.engine,
            cache_folder=arguments.cache_folder,
        )
        ocr_engine = PaddleOcrTextEngine(
            predictor,
            minimum_confidence=arguments.minimum_confidence,
        )
        rendered_pages = renderer.render(artifact.path)
        ocr_pages = tuple(
            ocr_engine.recognize(page) for page in rendered_pages
        )
        parsed = OcrParsedDocumentAssembler(
            SingleColumnOcrReadingOrder()
        ).assemble(artifact, ocr_pages)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    full_text = "\n".join(
        fragment.text for fragment in parsed.fragments
    )
    evaluation_payload: dict[str, object] = {}
    if arguments.expected_text_file is not None:
        expected_text = arguments.expected_text_file.read_text(
            encoding="utf-8"
        )
        evaluation = evaluate_ocr_text(expected_text, full_text)
        evaluation_payload = {
            "exact_match": evaluation.exact_match,
            "character_error_rate": round(
                evaluation.character_error_rate,
                6,
            ),
            "word_error_rate": round(
                evaluation.word_error_rate,
                6,
            ),
        }
    print(
        json.dumps(
            {
                "status": "passed",
                "page_count": len(rendered_pages),
                "fragment_count": len(parsed.fragments),
                "ocr_region_count": sum(
                    len(page.regions) for page in ocr_pages
                ),
                "engine_identity": ocr_engine.identity,
                "elapsed_seconds": round(
                    perf_counter() - started_at,
                    3,
                ),
                "text_preview": full_text[: arguments.preview_chars],
                **evaluation_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
