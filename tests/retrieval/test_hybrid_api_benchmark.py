import unittest

from app.application.retrieval.performance_metrics import (
    nearest_rank_percentile,
)


class HybridApiBenchmarkTest(unittest.TestCase):
    def test_uses_nearest_rank_percentile(self) -> None:
        values = [0.5, 0.1, 0.4, 0.2, 0.3]

        self.assertEqual(0.3, nearest_rank_percentile(values, 0.50))
        self.assertEqual(0.5, nearest_rank_percentile(values, 0.95))
        self.assertEqual(0.5, nearest_rank_percentile(values, 0.99))

    def test_rejects_empty_measurements(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            nearest_rank_percentile([], 0.95)


if __name__ == "__main__":
    unittest.main()
