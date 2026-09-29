import unittest
from pathlib import Path

from app.application.answering.grounded_answer_evaluation_json import (
    GroundedAnswerEvaluationDatasetError,
    decode_grounded_answer_evaluation_dataset,
    load_grounded_answer_evaluation_dataset,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "grounded-answer-smoke-v1.json"
)


class GroundedAnswerEvaluationJsonTest(unittest.TestCase):
    def test_loads_versioned_provisional_dataset(self) -> None:
        dataset = load_grounded_answer_evaluation_dataset(DATASET_PATH)

        self.assertEqual("grounded-answer-smoke-v1", dataset.dataset_id)
        self.assertEqual(1, dataset.dataset_version)
        self.assertEqual("provisional", dataset.review_status)
        self.assertEqual(4, len(dataset.cases))
        self.assertEqual("answered", dataset.cases[0].expected_status)
        self.assertEqual(
            "岗位要求",
            dataset.cases[0]
            .retrieved_chunks[0]
            .location.heading_path[0],
        )

    def test_rejects_duplicate_case_ids(self) -> None:
        case = {
            "case_id": "duplicate",
            "question": "问题",
            "expected_status": "insufficient_evidence",
            "retrieved_chunks": [],
        }
        payload = {
            "dataset_id": "invalid",
            "dataset_version": 1,
            "review_status": "provisional",
            "cases": [case, case],
        }

        with self.assertRaisesRegex(
            GroundedAnswerEvaluationDatasetError,
            "case IDs must be unique",
        ):
            decode_grounded_answer_evaluation_dataset(payload)

    def test_rejects_unknown_review_status(self) -> None:
        payload = {
            "dataset_id": "invalid",
            "dataset_version": 1,
            "review_status": "auto-approved",
            "cases": [
                {
                    "case_id": "one",
                    "question": "问题",
                    "expected_status": "insufficient_evidence",
                    "retrieved_chunks": [],
                }
            ],
        }

        with self.assertRaisesRegex(
            GroundedAnswerEvaluationDatasetError,
            "review_status",
        ):
            decode_grounded_answer_evaluation_dataset(payload)


if __name__ == "__main__":
    unittest.main()
