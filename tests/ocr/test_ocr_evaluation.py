import unittest

from app.application.ocr.ocr_evaluation import evaluate_ocr_text


class OcrTextEvaluationTest(unittest.TestCase):
    def test_exact_match_has_zero_error_rates(self) -> None:
        result = evaluate_ocr_text(
            "Position: AI Agent Intern",
            "Position: AI Agent Intern",
        )

        self.assertTrue(result.exact_match)
        self.assertEqual(0.0, result.character_error_rate)
        self.assertEqual(0.0, result.word_error_rate)

    def test_counts_ai_to_al_as_one_character_and_one_word_error(self) -> None:
        result = evaluate_ocr_text(
            "Position: AI Agent Intern",
            "Position: Al Agent Intern",
        )

        self.assertFalse(result.exact_match)
        self.assertAlmostEqual(
            1 / len("Position: AI Agent Intern"),
            result.character_error_rate,
        )
        self.assertEqual(0.25, result.word_error_rate)

    def test_normalizes_unicode_and_whitespace_before_comparison(self) -> None:
        result = evaluate_ocr_text(
            "Major:\u3000No restriction",
            "Major:   No\nrestriction",
        )

        self.assertTrue(result.exact_match)

    def test_rejects_a_blank_expected_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            evaluate_ocr_text("   ", "anything")


if __name__ == "__main__":
    unittest.main()
