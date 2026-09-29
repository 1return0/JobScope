from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from app.application.answering.answer_output_decoding import (
    AnswerOutputDecodingError,
)
from app.application.answering.grounded_answer_service import (
    GroundedAnswerResult,
    UngroundedModelAnswerError,
)
from app.application.answering.grounded_answering import AnswerStatus
from app.domain.documents.document_chunking import EvidenceChunk


class GroundedAnswerRunner(Protocol):
    def answer(
        self,
        question: str,
        *,
        retrieved_chunks: tuple[EvidenceChunk, ...],
    ) -> GroundedAnswerResult:
        ...


@dataclass(frozen=True, slots=True)
class GroundedAnswerEvaluationCase:
    case_id: str
    question: str
    retrieved_chunks: tuple[EvidenceChunk, ...]
    expected_status: AnswerStatus

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("answer evaluation case_id must not be blank")
        if not self.question.strip():
            raise ValueError("answer evaluation question must not be blank")
        if self.expected_status not in (
            "answered",
            "insufficient_evidence",
        ):
            raise ValueError("unsupported expected answer status")


@dataclass(frozen=True, slots=True)
class GroundedAnswerCaseEvaluation:
    case_id: str
    expected_status: AnswerStatus
    actual_status: AnswerStatus | None
    status_correct: bool
    grounding_valid: bool
    model_called: bool
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class GroundedAnswerEvaluationReport:
    case_count: int
    status_accuracy: float
    answer_accuracy: float
    refusal_accuracy: float
    grounding_pass_rate: float
    model_call_rate: float
    cases: tuple[GroundedAnswerCaseEvaluation, ...]


class GroundedAnswerEvaluator:
    def __init__(self, runner: GroundedAnswerRunner) -> None:
        self._runner = runner

    def evaluate(
        self,
        cases: tuple[GroundedAnswerEvaluationCase, ...],
    ) -> GroundedAnswerEvaluationReport:
        if not cases:
            raise ValueError("answer evaluation cases must not be empty")
        case_ids = tuple(case.case_id for case in cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("answer evaluation case IDs must be unique")

        evaluations = tuple(self._evaluate_case(case) for case in cases)
        answered = tuple(
            item
            for item in evaluations
            if item.expected_status == "answered"
        )
        refusals = tuple(
            item
            for item in evaluations
            if item.expected_status == "insufficient_evidence"
        )
        return GroundedAnswerEvaluationReport(
            case_count=len(evaluations),
            status_accuracy=_mean(
                item.status_correct for item in evaluations
            ),
            answer_accuracy=_mean(
                item.status_correct for item in answered
            ),
            refusal_accuracy=_mean(
                item.status_correct for item in refusals
            ),
            grounding_pass_rate=_mean(
                item.grounding_valid for item in evaluations
            ),
            model_call_rate=_mean(
                item.model_called for item in evaluations
            ),
            cases=evaluations,
        )

    def _evaluate_case(
        self,
        case: GroundedAnswerEvaluationCase,
    ) -> GroundedAnswerCaseEvaluation:
        try:
            result = self._runner.answer(
                case.question,
                retrieved_chunks=case.retrieved_chunks,
            )
        except AnswerOutputDecodingError as error:
            return _failed_case(
                case,
                failure_code=error.failure_code,
                model_called=True,
            )
        except UngroundedModelAnswerError:
            return _failed_case(
                case,
                failure_code="ungrounded-model-answer",
                model_called=True,
            )
        return GroundedAnswerCaseEvaluation(
            case_id=case.case_id,
            expected_status=case.expected_status,
            actual_status=result.draft.status,
            status_correct=result.draft.status == case.expected_status,
            grounding_valid=result.validation.valid,
            model_called=result.model_called,
            failure_code=None,
        )


def _failed_case(
    case: GroundedAnswerEvaluationCase,
    *,
    failure_code: str,
    model_called: bool,
) -> GroundedAnswerCaseEvaluation:
    return GroundedAnswerCaseEvaluation(
        case_id=case.case_id,
        expected_status=case.expected_status,
        actual_status=None,
        status_correct=False,
        grounding_valid=False,
        model_called=model_called,
        failure_code=failure_code,
    )


def _mean(values: Iterable[object]) -> float:
    materialized = tuple(values)
    if not materialized:
        return 0.0
    return sum(bool(value) for value in materialized) / len(materialized)
