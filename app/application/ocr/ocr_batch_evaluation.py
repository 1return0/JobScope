from __future__ import annotations

from dataclasses import dataclass

from app.application.ocr.ocr_evaluation import (
    OcrTextEvaluation,
    evaluate_ocr_text,
)


@dataclass(frozen=True, slots=True)
class OcrEvaluationCase:
    case_id: str
    expected_text: str
    actual_text: str

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("OCR evaluation case_id must not be blank")


@dataclass(frozen=True, slots=True)
class OcrCaseEvaluation:
    case_id: str
    text_evaluation: OcrTextEvaluation


@dataclass(frozen=True, slots=True)
class OcrBatchEvaluation:
    cases: tuple[OcrCaseEvaluation, ...]
    exact_match_rate: float
    macro_character_error_rate: float
    micro_character_error_rate: float
    macro_word_error_rate: float
    micro_word_error_rate: float


def evaluate_ocr_batch(
    cases: tuple[OcrEvaluationCase, ...],
) -> OcrBatchEvaluation:
    if not cases:
        raise ValueError("OCR evaluation batch must not be empty")
    case_ids = tuple(case.case_id.strip() for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("OCR evaluation case_id values must be unique")

    evaluations = tuple(
        OcrCaseEvaluation(
            case_id=case.case_id.strip(),
            text_evaluation=evaluate_ocr_text(
                case.expected_text,
                case.actual_text,
            ),
        )
        for case in cases
    )
    case_count = len(evaluations)
    total_character_edits = sum(
        case.text_evaluation.character_edit_count
        for case in evaluations
    )
    total_expected_characters = sum(
        case.text_evaluation.expected_character_count
        for case in evaluations
    )
    total_word_edits = sum(
        case.text_evaluation.word_edit_count
        for case in evaluations
    )
    total_expected_words = sum(
        case.text_evaluation.expected_word_count
        for case in evaluations
    )
    return OcrBatchEvaluation(
        cases=evaluations,
        exact_match_rate=(
            sum(case.text_evaluation.exact_match for case in evaluations)
            / case_count
        ),
        macro_character_error_rate=(
            sum(
                case.text_evaluation.character_error_rate
                for case in evaluations
            )
            / case_count
        ),
        micro_character_error_rate=(
            total_character_edits / total_expected_characters
        ),
        macro_word_error_rate=(
            sum(
                case.text_evaluation.word_error_rate
                for case in evaluations
            )
            / case_count
        ),
        micro_word_error_rate=total_word_edits / total_expected_words,
    )
