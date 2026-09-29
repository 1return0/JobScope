from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.application.ocr.ocr_batch_evaluation import (
    OcrBatchEvaluation,
    OcrEvaluationCase,
    evaluate_ocr_batch,
)


class OcrDocumentTextRecognizer(Protocol):
    @property
    def identity(self) -> str:
        ...

    def recognize_text(self, artifact_path: Path) -> str:
        ...


@dataclass(frozen=True, slots=True)
class OcrDatasetExperimentCase:
    case_id: str
    artifact_path: Path
    expected_text: str


@dataclass(frozen=True, slots=True)
class OcrDatasetExperimentReport:
    dataset_id: str
    dataset_sha256: str
    recognizer_identity: str
    evaluation: OcrBatchEvaluation


class OcrDatasetExperimentRunner:
    def __init__(self, recognizer: OcrDocumentTextRecognizer) -> None:
        if not recognizer.identity.strip():
            raise ValueError("OCR recognizer identity must not be blank")
        self._recognizer = recognizer

    def run(
        self,
        *,
        dataset_id: str,
        dataset_sha256: str,
        cases: tuple[OcrDatasetExperimentCase, ...],
    ) -> OcrDatasetExperimentReport:
        if not dataset_id.strip():
            raise ValueError("OCR experiment dataset_id must not be blank")
        if len(dataset_sha256) != 64:
            raise ValueError(
                "OCR experiment dataset_sha256 must be a SHA-256 digest"
            )
        if not cases:
            raise ValueError("OCR experiment must contain cases")

        evaluations: list[OcrEvaluationCase] = []
        for case in cases:
            actual_text = self._recognizer.recognize_text(
                case.artifact_path
            )
            evaluations.append(
                OcrEvaluationCase(
                    case_id=case.case_id,
                    expected_text=case.expected_text,
                    actual_text=actual_text,
                )
            )

        return OcrDatasetExperimentReport(
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            recognizer_identity=self._recognizer.identity,
            evaluation=evaluate_ocr_batch(tuple(evaluations)),
        )
