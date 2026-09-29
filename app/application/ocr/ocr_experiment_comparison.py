from __future__ import annotations

from dataclasses import dataclass

from app.application.ocr.ocr_dataset_experiment import (
    OcrDatasetExperimentReport,
)


@dataclass(frozen=True, slots=True)
class OcrExperimentComparison:
    dataset_id: str
    dataset_sha256: str
    baseline_recognizer_identity: str
    candidate_recognizer_identity: str
    exact_match_rate_improvement: float
    macro_character_error_rate_reduction: float
    micro_character_error_rate_reduction: float
    macro_word_error_rate_reduction: float
    micro_word_error_rate_reduction: float


def compare_ocr_experiments(
    baseline: OcrDatasetExperimentReport,
    candidate: OcrDatasetExperimentReport,
) -> OcrExperimentComparison:
    if (
        baseline.dataset_id != candidate.dataset_id
        or baseline.dataset_sha256 != candidate.dataset_sha256
    ):
        raise ValueError(
            "OCR experiments must use the same frozen dataset"
        )

    baseline_case_ids = tuple(
        case.case_id for case in baseline.evaluation.cases
    )
    candidate_case_ids = tuple(
        case.case_id for case in candidate.evaluation.cases
    )
    if baseline_case_ids != candidate_case_ids:
        raise ValueError(
            "OCR experiments must evaluate the same cases in the same order"
        )

    baseline_metrics = baseline.evaluation
    candidate_metrics = candidate.evaluation
    return OcrExperimentComparison(
        dataset_id=baseline.dataset_id,
        dataset_sha256=baseline.dataset_sha256,
        baseline_recognizer_identity=baseline.recognizer_identity,
        candidate_recognizer_identity=candidate.recognizer_identity,
        exact_match_rate_improvement=(
            candidate_metrics.exact_match_rate
            - baseline_metrics.exact_match_rate
        ),
        macro_character_error_rate_reduction=(
            baseline_metrics.macro_character_error_rate
            - candidate_metrics.macro_character_error_rate
        ),
        micro_character_error_rate_reduction=(
            baseline_metrics.micro_character_error_rate
            - candidate_metrics.micro_character_error_rate
        ),
        macro_word_error_rate_reduction=(
            baseline_metrics.macro_word_error_rate
            - candidate_metrics.macro_word_error_rate
        ),
        micro_word_error_rate_reduction=(
            baseline_metrics.micro_word_error_rate
            - candidate_metrics.micro_word_error_rate
        ),
    )
