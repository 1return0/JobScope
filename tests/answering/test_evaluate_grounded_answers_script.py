import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from app.application.answering.grounded_answer_evaluation import (
    GroundedAnswerCaseEvaluation,
    GroundedAnswerEvaluationCase,
    GroundedAnswerEvaluationReport,
)
from app.application.answering.grounded_answer_evaluation_json import (
    GroundedAnswerEvaluationDataset,
)
from scripts.answering.evaluate_grounded_answers import (
    save_report,
    serialize_grounded_answer_evaluation_report,
)


class EvaluateGroundedAnswersScriptTest(unittest.TestCase):
    def test_serializes_dataset_identity_metrics_and_limitations(self) -> None:
        dataset = GroundedAnswerEvaluationDataset(
            dataset_id="test-dataset",
            dataset_version=1,
            review_status="provisional",
            cases=(
                GroundedAnswerEvaluationCase(
                    case_id="one",
                    question="问题",
                    retrieved_chunks=(),
                    expected_status="insufficient_evidence",
                ),
            ),
        )
        report = GroundedAnswerEvaluationReport(
            case_count=1,
            status_accuracy=1.0,
            answer_accuracy=0.0,
            refusal_accuracy=1.0,
            grounding_pass_rate=1.0,
            model_call_rate=0.0,
            cases=(
                GroundedAnswerCaseEvaluation(
                    case_id="one",
                    expected_status="insufficient_evidence",
                    actual_status="insufficient_evidence",
                    status_correct=True,
                    grounding_valid=True,
                    model_called=False,
                    failure_code=None,
                ),
            ),
        )

        payload = serialize_grounded_answer_evaluation_report(
            dataset,
            report,
            model_name="test-model",
            dataset_sha256="a" * 64,
            prompt_version="grounded-answer-v2",
            executed_at=datetime(
                2026,
                8,
                10,
                12,
                0,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual("test-dataset", payload["dataset_id"])
        self.assertEqual("provisional", payload["review_status"])
        self.assertEqual("test-model", payload["model_name"])
        self.assertEqual("a" * 64, payload["dataset_sha256"])
        self.assertEqual("grounded-answer-v2", payload["prompt_version"])
        self.assertEqual(1.0, payload["metrics"]["refusal_accuracy"])
        self.assertIn("provisional", payload["limitations"][0])

    def test_report_file_cannot_silently_overwrite_history(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            save_report(output, '{"run":1}')

            with self.assertRaises(FileExistsError):
                save_report(output, '{"run":2}')

            self.assertEqual('{"run":1}\n', output.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
