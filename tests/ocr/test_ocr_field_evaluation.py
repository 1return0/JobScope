import unittest

from app.application.ocr.ocr_field_evaluation import (
    OcrFieldJudgment,
    evaluate_ocr_fields,
)


class OcrFieldEvaluationTest(unittest.TestCase):
    def test_separates_overall_and_critical_field_accuracy(self) -> None:
        report = evaluate_ocr_fields(
            (
                OcrFieldJudgment(
                    "case-1",
                    "job_title",
                    "AI Agent Intern",
                    "Al Agent Intern",
                    critical=True,
                ),
                OcrFieldJudgment(
                    "case-1",
                    "deadline",
                    "2026-09-30",
                    "2026-09-30",
                    critical=True,
                ),
                OcrFieldJudgment(
                    "case-1",
                    "footer",
                    "Page 1",
                    "Page 1",
                    critical=False,
                ),
            )
        )

        self.assertAlmostEqual(2 / 3, report.overall_exact_match_rate)
        self.assertEqual(0.5, report.critical_exact_match_rate)

    def test_reports_accuracy_for_each_field_name(self) -> None:
        report = evaluate_ocr_fields(
            (
                OcrFieldJudgment("a", "deadline", "June", "June"),
                OcrFieldJudgment("b", "deadline", "July", "Juiy"),
                OcrFieldJudgment("a", "location", "上海", "上海"),
            )
        )

        metrics = {
            metric.field_name: metric
            for metric in report.metrics_by_field
        }
        self.assertEqual(0.5, metrics["deadline"].exact_match_rate)
        self.assertEqual(1.0, metrics["location"].exact_match_rate)

    def test_rejects_duplicate_document_field_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            evaluate_ocr_fields(
                (
                    OcrFieldJudgment("same", "deadline", "A", "A"),
                    OcrFieldJudgment("same", "deadline", "B", "B"),
                )
            )


if __name__ == "__main__":
    unittest.main()
