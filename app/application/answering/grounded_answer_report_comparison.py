from __future__ import annotations

from dataclasses import dataclass

from app.application.answering.grounded_answer_evaluation import (
    GroundedAnswerCaseEvaluation,
    GroundedAnswerEvaluationReport,
)


@dataclass(frozen=True, slots=True)
class GroundedAnswerCaseComparison:
    case_id: str
    baseline_status_correct: bool
    candidate_status_correct: bool
    status_regressed: bool
    status_improved: bool
    baseline_grounding_valid: bool
    candidate_grounding_valid: bool
    grounding_regressed: bool


@dataclass(frozen=True, slots=True)
class GroundedAnswerReportComparison:
    case_count: int
    status_accuracy_delta: float
    answer_accuracy_delta: float
    refusal_accuracy_delta: float
    grounding_pass_rate_delta: float
    status_regression_case_ids: tuple[str, ...]
    status_improvement_case_ids: tuple[str, ...]
    grounding_regression_case_ids: tuple[str, ...]
    cases: tuple[GroundedAnswerCaseComparison, ...]


def compare_grounded_answer_reports(
    baseline: GroundedAnswerEvaluationReport,
    candidate: GroundedAnswerEvaluationReport,
) -> GroundedAnswerReportComparison:
    baseline_by_id = _index_cases(baseline.cases)
    candidate_by_id = _index_cases(candidate.cases)
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("answer reports contain different case IDs")

    comparisons = tuple(
        _compare_case(
            baseline_by_id[case_id],
            candidate_by_id[case_id],
        )
        for case_id in baseline_by_id
    )
    return GroundedAnswerReportComparison(
        case_count=len(comparisons),
        status_accuracy_delta=(
            candidate.status_accuracy - baseline.status_accuracy
        ),
        answer_accuracy_delta=(
            candidate.answer_accuracy - baseline.answer_accuracy
        ),
        refusal_accuracy_delta=(
            candidate.refusal_accuracy - baseline.refusal_accuracy
        ),
        grounding_pass_rate_delta=(
            candidate.grounding_pass_rate
            - baseline.grounding_pass_rate
        ),
        status_regression_case_ids=tuple(
            item.case_id for item in comparisons if item.status_regressed
        ),
        status_improvement_case_ids=tuple(
            item.case_id for item in comparisons if item.status_improved
        ),
        grounding_regression_case_ids=tuple(
            item.case_id
            for item in comparisons
            if item.grounding_regressed
        ),
        cases=comparisons,
    )


def _index_cases(
    cases: tuple[GroundedAnswerCaseEvaluation, ...],
) -> dict[str, GroundedAnswerCaseEvaluation]:
    indexed = {case.case_id: case for case in cases}
    if len(indexed) != len(cases):
        raise ValueError("answer report contains duplicate case IDs")
    return indexed


def _compare_case(
    baseline: GroundedAnswerCaseEvaluation,
    candidate: GroundedAnswerCaseEvaluation,
) -> GroundedAnswerCaseComparison:
    if baseline.expected_status != candidate.expected_status:
        raise ValueError(
            "answer reports contain different expected statuses"
        )
    return GroundedAnswerCaseComparison(
        case_id=baseline.case_id,
        baseline_status_correct=baseline.status_correct,
        candidate_status_correct=candidate.status_correct,
        status_regressed=(
            baseline.status_correct and not candidate.status_correct
        ),
        status_improved=(
            not baseline.status_correct and candidate.status_correct
        ),
        baseline_grounding_valid=baseline.grounding_valid,
        candidate_grounding_valid=candidate.grounding_valid,
        grounding_regressed=(
            baseline.grounding_valid and not candidate.grounding_valid
        ),
    )
