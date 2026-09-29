import unittest

from app.application.ocr.ocr_batch_evaluation import (
    OcrEvaluationCase,
    evaluate_ocr_batch,
)


class OcrBatchEvaluationTest(unittest.TestCase):
    def test_reports_exact_macro_and_micro_metrics(self) -> None:
        result = evaluate_ocr_batch(
            (
                OcrEvaluationCase(
                    case_id="short-error",
                    expected_text="AI",
                    actual_text="Al",
                ),
                OcrEvaluationCase(
                    case_id="long-correct",
                    expected_text="Campus recruitment information",
                    actual_text="Campus recruitment information",
                ),
            )
        )

        self.assertEqual(0.5, result.exact_match_rate)
        self.assertEqual(0.25, result.macro_character_error_rate)
        self.assertLess(
            result.micro_character_error_rate,
            result.macro_character_error_rate,
        )
        self.assertEqual(0.5, result.macro_word_error_rate)
        self.assertEqual(0.25, result.micro_word_error_rate)

    def test_preserves_case_level_failures_for_diagnosis(self) -> None:
        result = evaluate_ocr_batch(
            (
                OcrEvaluationCase("a", "AI", "Al"),
                OcrEvaluationCase("b", "RAG", "RAG"),
            )
        )

        self.assertEqual(
            ("a", "b"),
            tuple(case.case_id for case in result.cases),
        )
        self.assertFalse(result.cases[0].text_evaluation.exact_match)

    def test_rejects_empty_or_duplicate_case_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            evaluate_ocr_batch(())

        with self.assertRaisesRegex(ValueError, "must be unique"):
            evaluate_ocr_batch(
                (
                    OcrEvaluationCase("same", "A", "A"),
                    OcrEvaluationCase("same", "B", "B"),
                )
            )


if __name__ == "__main__":
    unittest.main()
