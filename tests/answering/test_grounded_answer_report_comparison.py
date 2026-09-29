import unittest

from app.application.answering.grounded_answer_evaluation import (
    GroundedAnswerCaseEvaluation,
    GroundedAnswerEvaluationReport,
)
from app.application.answering.grounded_answer_report_comparison import (
    compare_grounded_answer_reports,
)


class GroundedAnswerReportComparisonTest(unittest.TestCase):
    def test_finds_case_regression_when_aggregate_score_is_unchanged(
        self,
    ) -> None:
        baseline = self._report(
            self._case("stable-before", status_correct=True),
            self._case("improved-after", status_correct=False),
        )
        candidate = self._report(
            self._case("stable-before", status_correct=False),
            self._case("improved-after", status_correct=True),
        )

        comparison = compare_grounded_answer_reports(
            baseline,
            candidate,
        )

        self.assertEqual(0.0, comparison.status_accuracy_delta)
        self.assertEqual(
            ("stable-before",),
            comparison.status_regression_case_ids,
        )
        self.assertEqual(
            ("improved-after",),
            comparison.status_improvement_case_ids,
        )

    def test_finds_grounding_regression(self) -> None:
        baseline = self._report(
            self._case("case-a", grounding_valid=True),
        )
        candidate = self._report(
            self._case("case-a", grounding_valid=False),
        )

        comparison = compare_grounded_answer_reports(
            baseline,
            candidate,
        )

        self.assertEqual(
            ("case-a",),
            comparison.grounding_regression_case_ids,
        )

    def test_rejects_different_case_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "different case IDs"):
            compare_grounded_answer_reports(
                self._report(self._case("case-a")),
                self._report(self._case("case-b")),
            )

    @staticmethod
    def _case(
        case_id: str,
        *,
        status_correct: bool = True,
        grounding_valid: bool = True,
    ) -> GroundedAnswerCaseEvaluation:
        return GroundedAnswerCaseEvaluation(
            case_id=case_id,
            expected_status="answered",
            actual_status=(
                "answered" if status_correct else "insufficient_evidence"
            ),
            status_correct=status_correct,
            grounding_valid=grounding_valid,
            model_called=True,
            failure_code=None,
        )

    @staticmethod
    def _report(
        *cases: GroundedAnswerCaseEvaluation,
    ) -> GroundedAnswerEvaluationReport:
        status_accuracy = sum(
            case.status_correct for case in cases
        ) / len(cases)
        grounding_pass_rate = sum(
            case.grounding_valid for case in cases
        ) / len(cases)
        return GroundedAnswerEvaluationReport(
            case_count=len(cases),
            status_accuracy=status_accuracy,
            answer_accuracy=status_accuracy,
            refusal_accuracy=0.0,
            grounding_pass_rate=grounding_pass_rate,
            model_call_rate=1.0,
            cases=cases,
        )


if __name__ == "__main__":
    unittest.main()
