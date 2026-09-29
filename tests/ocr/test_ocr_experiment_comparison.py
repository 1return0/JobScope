import unittest

from app.application.ocr.ocr_batch_evaluation import (
    OcrEvaluationCase,
    evaluate_ocr_batch,
)
from app.application.ocr.ocr_dataset_experiment import (
    OcrDatasetExperimentReport,
)
from app.application.ocr.ocr_experiment_comparison import (
    compare_ocr_experiments,
)


def _report(
    *,
    dataset_sha256: str = "a" * 64,
    identity: str,
    actual_text: str,
) -> OcrDatasetExperimentReport:
    return OcrDatasetExperimentReport(
        dataset_id="jobscope-ocr-v1",
        dataset_sha256=dataset_sha256,
        recognizer_identity=identity,
        evaluation=evaluate_ocr_batch(
            (OcrEvaluationCase("case-a", "AI Agent", actual_text),)
        ),
    )


class OcrExperimentComparisonTest(unittest.TestCase):
    def test_reports_improvement_on_the_same_frozen_dataset(self) -> None:
        comparison = compare_ocr_experiments(
            _report(identity="dpi=200", actual_text="Al Agent"),
            _report(identity="dpi=300", actual_text="AI Agent"),
        )

        self.assertEqual(1.0, comparison.exact_match_rate_improvement)
        self.assertGreater(
            comparison.micro_character_error_rate_reduction,
            0.0,
        )
        self.assertEqual("dpi=200", comparison.baseline_recognizer_identity)
        self.assertEqual("dpi=300", comparison.candidate_recognizer_identity)

    def test_rejects_comparison_when_dataset_content_changed(self) -> None:
        with self.assertRaisesRegex(ValueError, "same frozen dataset"):
            compare_ocr_experiments(
                _report(
                    dataset_sha256="a" * 64,
                    identity="dpi=200",
                    actual_text="AI Agent",
                ),
                _report(
                    dataset_sha256="b" * 64,
                    identity="dpi=300",
                    actual_text="AI Agent",
                ),
            )


if __name__ == "__main__":
    unittest.main()
