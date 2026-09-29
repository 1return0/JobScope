import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from app.application.ocr.ocr_batch_evaluation import (
    OcrEvaluationCase,
    evaluate_ocr_batch,
)
from app.application.ocr.ocr_dataset_experiment import (
    OcrDatasetExperimentReport,
)
from app.infrastructure.reports.ocr_experiment_report_json import (
    save_ocr_experiment_report,
    serialize_ocr_experiment_report,
)


class OcrExperimentReportJsonTest(unittest.TestCase):
    def test_serializes_reproducible_identity_metrics_and_case_details(
        self,
    ) -> None:
        report = OcrDatasetExperimentReport(
            dataset_id="jobscope-ocr-v1",
            dataset_sha256="a" * 64,
            recognizer_identity="paddleocr;revision=1;dpi=300",
            evaluation=evaluate_ocr_batch(
                (OcrEvaluationCase("case-a", "AI Agent", "Al Agent"),)
            ),
        )

        payload = serialize_ocr_experiment_report(
            report,
            executed_at=datetime(
                2026,
                8,
                15,
                12,
                0,
                tzinfo=timezone.utc,
            ),
        )

        self.assertEqual("a" * 64, payload["dataset_sha256"])
        self.assertEqual(
            "paddleocr;revision=1;dpi=300",
            payload["recognizer_identity"],
        )
        self.assertEqual("AI Agent", payload["cases"][0]["normalized_expected"])
        self.assertEqual("Al Agent", payload["cases"][0]["normalized_actual"])

    def test_saved_report_cannot_silently_overwrite_history(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "ocr-report.json"
            save_ocr_experiment_report(output_path, {"run": 1})

            with self.assertRaises(FileExistsError):
                save_ocr_experiment_report(output_path, {"run": 2})

            self.assertEqual(
                {"run": 1},
                json.loads(output_path.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
