import unittest

from app.application.answering.grounded_answer_evaluation import (
    GroundedAnswerEvaluationCase,
    GroundedAnswerEvaluator,
)
from app.application.answering.grounded_answer_service import GroundedAnswerResult
from app.application.answering.grounded_answering import (
    CitationValidationReport,
    GroundedAnswerDraft,
)


class _AnswerRunner:
    def __init__(self, results: dict[str, GroundedAnswerResult]) -> None:
        self._results = results

    def answer(self, question: str, *, retrieved_chunks):
        return self._results[question]


def _result(status: str, *, model_called: bool) -> GroundedAnswerResult:
    if status == "answered":
        from app.application.answering.grounded_answering import (
            EvidenceCitation,
            GroundedClaim,
        )

        draft = GroundedAnswerDraft(
            status="answered",
            claims=(
                GroundedClaim(
                    text="该岗位专业不限。",
                    citations=(
                        EvidenceCitation(
                            evidence_id="ev_" + "1" * 64,
                            quoted_text="专业不限",
                        ),
                    ),
                ),
            ),
        )
    else:
        draft = GroundedAnswerDraft(
            status="insufficient_evidence",
            claims=(),
            fallback_message="当前证据不足。",
        )
    return GroundedAnswerResult(
        draft=draft,
        validation=CitationValidationReport(
            valid=True,
            checked_claim_count=len(draft.claims),
            checked_citation_count=sum(
                len(claim.citations) for claim in draft.claims
            ),
            issues=(),
        ),
        model_called=model_called,
    )


class GroundedAnswerEvaluatorTest(unittest.TestCase):
    def test_reports_answer_refusal_and_model_call_metrics(self) -> None:
        runner = _AnswerRunner(
            {
                "岗位限制专业吗？": _result(
                    "answered",
                    model_called=True,
                ),
                "截止日期是什么？": _result(
                    "insufficient_evidence",
                    model_called=False,
                ),
            }
        )
        cases = (
            GroundedAnswerEvaluationCase(
                case_id="answerable",
                question="岗位限制专业吗？",
                retrieved_chunks=(),
                expected_status="answered",
            ),
            GroundedAnswerEvaluationCase(
                case_id="must-refuse",
                question="截止日期是什么？",
                retrieved_chunks=(),
                expected_status="insufficient_evidence",
            ),
        )

        report = GroundedAnswerEvaluator(runner).evaluate(cases)

        self.assertEqual(2, report.case_count)
        self.assertEqual(1.0, report.status_accuracy)
        self.assertEqual(1.0, report.answer_accuracy)
        self.assertEqual(1.0, report.refusal_accuracy)
        self.assertEqual(1.0, report.grounding_pass_rate)
        self.assertEqual(0.5, report.model_call_rate)

    def test_wrong_refusal_status_reduces_only_expected_metrics(self) -> None:
        runner = _AnswerRunner(
            {
                "有证据的问题": _result("answered", model_called=True),
                "必须拒答的问题": _result("answered", model_called=True),
            }
        )
        cases = (
            GroundedAnswerEvaluationCase(
                case_id="answerable",
                question="有证据的问题",
                retrieved_chunks=(),
                expected_status="answered",
            ),
            GroundedAnswerEvaluationCase(
                case_id="must-refuse",
                question="必须拒答的问题",
                retrieved_chunks=(),
                expected_status="insufficient_evidence",
            ),
        )

        report = GroundedAnswerEvaluator(runner).evaluate(cases)

        self.assertEqual(0.5, report.status_accuracy)
        self.assertEqual(1.0, report.answer_accuracy)
        self.assertEqual(0.0, report.refusal_accuracy)

    def test_rejects_duplicate_case_ids(self) -> None:
        case = GroundedAnswerEvaluationCase(
            case_id="duplicate",
            question="问题",
            retrieved_chunks=(),
            expected_status="insufficient_evidence",
        )

        with self.assertRaisesRegex(ValueError, "must be unique"):
            GroundedAnswerEvaluator(_AnswerRunner({})).evaluate(
                (case, case)
            )


if __name__ == "__main__":
    unittest.main()
