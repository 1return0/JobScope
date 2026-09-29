import unittest
from pathlib import Path

from app.application.ocr.ocr_dataset_experiment import (
    OcrDatasetExperimentCase,
    OcrDatasetExperimentRunner,
)


class _Recognizer:
    identity = "paddleocr;model=test;revision=1"

    def __init__(self, results: dict[Path, str | Exception]) -> None:
        self._results = results
        self.calls: list[Path] = []

    def recognize_text(self, artifact_path: Path) -> str:
        self.calls.append(artifact_path)
        result = self._results[artifact_path]
        if isinstance(result, Exception):
            raise result
        return result


class OcrDatasetExperimentRunnerTest(unittest.TestCase):
    def test_runs_every_case_and_preserves_experiment_identity(self) -> None:
        first_path = Path("first.pdf")
        second_path = Path("second.pdf")
        recognizer = _Recognizer(
            {
                first_path: "Al Agent",
                second_path: "Shanghai",
            }
        )
        runner = OcrDatasetExperimentRunner(recognizer)

        report = runner.run(
            dataset_id="jobscope-ocr-v1",
            dataset_sha256="a" * 64,
            cases=(
                OcrDatasetExperimentCase(
                    "first",
                    first_path,
                    "AI Agent",
                ),
                OcrDatasetExperimentCase(
                    "second",
                    second_path,
                    "Shanghai",
                ),
            ),
        )

        self.assertEqual([first_path, second_path], recognizer.calls)
        self.assertEqual("a" * 64, report.dataset_sha256)
        self.assertEqual(recognizer.identity, report.recognizer_identity)
        self.assertEqual(0.5, report.evaluation.exact_match_rate)

    def test_does_not_publish_partial_metrics_after_technical_failure(
        self,
    ) -> None:
        first_path = Path("first.pdf")
        second_path = Path("second.pdf")
        runner = OcrDatasetExperimentRunner(
            _Recognizer(
                {
                    first_path: "correct",
                    second_path: RuntimeError("model failed"),
                }
            )
        )

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            runner.run(
                dataset_id="jobscope-ocr-v1",
                dataset_sha256="a" * 64,
                cases=(
                    OcrDatasetExperimentCase(
                        "first",
                        first_path,
                        "correct",
                    ),
                    OcrDatasetExperimentCase(
                        "second",
                        second_path,
                        "expected",
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
