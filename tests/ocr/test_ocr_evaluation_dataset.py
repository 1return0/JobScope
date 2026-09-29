import unittest

from app.application.ocr.ocr_evaluation_dataset import (
    OcrEvaluationDataset,
    OcrGroundTruthCase,
    OcrGroundTruthField,
)


def _case(*, expected_hash: str = "b" * 64) -> OcrGroundTruthCase:
    return OcrGroundTruthCase(
        case_id="clean-en-single-column",
        artifact_reference="data/ocr/clean-en.pdf",
        artifact_sha256="a" * 64,
        expected_text_reference="data/ocr/clean-en.expected.txt",
        expected_text_sha256=expected_hash,
        tags=("synthetic", "english", "single-column"),
        fields=(
            OcrGroundTruthField(
                field_name="job_title",
                expected_value="AI Agent Intern",
            ),
        ),
    )


class OcrEvaluationDatasetTest(unittest.TestCase):
    def test_same_dataset_content_has_a_stable_fingerprint(self) -> None:
        first = OcrEvaluationDataset(
            "jobscope-ocr-v1",
            1,
            "reviewed",
            (_case(),),
        )
        second = OcrEvaluationDataset(
            "jobscope-ocr-v1",
            1,
            "reviewed",
            (_case(),),
        )

        self.assertEqual(first.dataset_sha256, second.dataset_sha256)

    def test_ground_truth_change_produces_a_new_fingerprint(self) -> None:
        original = OcrEvaluationDataset(
            "jobscope-ocr-v1",
            1,
            "reviewed",
            (_case(),),
        )
        changed = OcrEvaluationDataset(
            "jobscope-ocr-v1",
            1,
            "reviewed",
            (_case(expected_hash="c" * 64),),
        )

        self.assertNotEqual(
            original.dataset_sha256,
            changed.dataset_sha256,
        )

    def test_rejects_duplicate_case_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            OcrEvaluationDataset(
                "jobscope-ocr-v1",
                1,
                "provisional",
                (_case(), _case()),
            )


if __name__ == "__main__":
    unittest.main()
