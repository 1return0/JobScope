from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.application.ocr.ocr_dataset_experiment import (
    OcrDatasetExperimentCase,
    OcrDatasetExperimentRunner,
)
from app.application.ocr.pdf_ocr_parsing import (
    OcrParsedDocumentAssembler,
    SingleColumnOcrReadingOrder,
)
from app.infrastructure.reports.ocr_evaluation_dataset_json import (
    load_ocr_evaluation_dataset,
    verify_ocr_evaluation_dataset_files,
)
from app.infrastructure.reports.ocr_experiment_report_json import (
    save_ocr_experiment_report,
    serialize_ocr_experiment_report,
)
from app.infrastructure.ocr.paddle_ocr_engine import PaddleOcrTextEngine
from app.infrastructure.ocr.paddle_ocr_pipeline_predictor import (
    build_paddle_ocr_pipeline_predictor,
)
from app.infrastructure.ocr.pdf_ocr_text_recognizer import (
    PdfOcrTextRecognizer,
)
from app.infrastructure.ocr.pdfium_page_renderer import (
    PdfiumPageRendererConfig,
    build_pdfium_page_renderer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate real OCR against a frozen JobScope dataset."
    )
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("."))
    parser.add_argument("--engine", default="transformers")
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=Path(".model-cache") / "paddlex",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    dataset = load_ocr_evaluation_dataset(arguments.dataset_path)
    verified_cases = verify_ocr_evaluation_dataset_files(
        dataset,
        artifact_root=arguments.artifact_root,
    )
    predictor = build_paddle_ocr_pipeline_predictor(
        engine=arguments.engine,
        cache_folder=arguments.cache_folder,
    )
    ocr_engine = PaddleOcrTextEngine(
        predictor,
        minimum_confidence=arguments.minimum_confidence,
    )
    recognizer = PdfOcrTextRecognizer(
        build_pdfium_page_renderer(
            PdfiumPageRendererConfig(
                dpi=arguments.dpi,
                max_pages=arguments.max_pages,
            )
        ),
        ocr_engine,
        OcrParsedDocumentAssembler(
            SingleColumnOcrReadingOrder()
        ),
        configuration_identity=(
            f"engine={arguments.engine};dpi={arguments.dpi};"
            f"minimum-confidence={arguments.minimum_confidence}"
        ),
    )
    report = OcrDatasetExperimentRunner(recognizer).run(
        dataset_id=dataset.dataset_id,
        dataset_sha256=dataset.dataset_sha256,
        cases=tuple(
            OcrDatasetExperimentCase(
                case_id=item.case.case_id,
                artifact_path=item.artifact_path,
                expected_text=item.expected_text_path.read_text(
                    encoding="utf-8"
                ),
            )
            for item in verified_cases
        ),
    )
    payload = serialize_ocr_experiment_report(
        report,
        executed_at=datetime.now(timezone.utc),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if arguments.output is not None:
        save_ocr_experiment_report(arguments.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
