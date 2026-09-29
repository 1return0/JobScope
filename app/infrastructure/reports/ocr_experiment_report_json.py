from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.application.ocr.ocr_dataset_experiment import (
    OcrDatasetExperimentReport,
)


def serialize_ocr_experiment_report(
    report: OcrDatasetExperimentReport,
    *,
    executed_at: datetime,
) -> dict[str, object]:
    if executed_at.tzinfo is None:
        raise ValueError("OCR experiment executed_at must have timezone")

    evaluation = report.evaluation
    return {
        "dataset_id": report.dataset_id,
        "dataset_sha256": report.dataset_sha256,
        "recognizer_identity": report.recognizer_identity,
        "executed_at": executed_at.isoformat(),
        "case_count": len(evaluation.cases),
        "metrics": {
            "exact_match_rate": evaluation.exact_match_rate,
            "macro_character_error_rate": (
                evaluation.macro_character_error_rate
            ),
            "micro_character_error_rate": (
                evaluation.micro_character_error_rate
            ),
            "macro_word_error_rate": evaluation.macro_word_error_rate,
            "micro_word_error_rate": evaluation.micro_word_error_rate,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "normalized_expected": (
                    case.text_evaluation.normalized_expected
                ),
                "normalized_actual": (
                    case.text_evaluation.normalized_actual
                ),
                "exact_match": case.text_evaluation.exact_match,
                "character_edit_count": (
                    case.text_evaluation.character_edit_count
                ),
                "character_error_rate": (
                    case.text_evaluation.character_error_rate
                ),
                "word_edit_count": case.text_evaluation.word_edit_count,
                "word_error_rate": case.text_evaluation.word_error_rate,
            }
            for case in evaluation.cases
        ],
    }


def save_ocr_experiment_report(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("x", encoding="utf-8", newline="\n") as report_file:
        report_file.write(rendered)
        report_file.write("\n")
