import unittest
from datetime import date

from app.application.jobs.job_record_normalization import (
    ApplicationDeadlineNormalizer,
)
from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    JobFieldCitation,
)


def _fact(value: str | None) -> EvidenceBackedJobFact:
    if value is None:
        return EvidenceBackedJobFact(None)
    return EvidenceBackedJobFact(
        value,
        (JobFieldCitation("ev-1", value),),
    )


class ApplicationDeadlineNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = ApplicationDeadlineNormalizer()

    def test_normalizes_supported_chinese_date_and_keeps_raw_fact(self) -> None:
        raw_fact = _fact("2026年9月30日")

        result = self.normalizer.normalize(raw_fact)

        self.assertEqual(date(2026, 9, 30), result.normalized_date)
        self.assertEqual("normalized", result.status)
        self.assertIs(raw_fact, result.raw_fact)

    def test_unknown_deadline_remains_not_stated(self) -> None:
        result = self.normalizer.normalize(_fact(None))

        self.assertEqual("not_stated", result.status)
        self.assertIsNone(result.normalized_date)

    def test_ambiguous_expression_requires_manual_review(self) -> None:
        result = self.normalizer.normalize(_fact("招满即止"))

        self.assertEqual("manual_review_required", result.status)
        self.assertEqual(
            "unsupported-date-expression",
            result.reason_code,
        )

    def test_invalid_calendar_date_requires_manual_review(self) -> None:
        result = self.normalizer.normalize(_fact("2026-02-30"))

        self.assertEqual("manual_review_required", result.status)
        self.assertEqual("invalid-calendar-date", result.reason_code)


if __name__ == "__main__":
    unittest.main()
