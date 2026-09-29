import unittest

from app.domain.jobs.recruitment_type import (
    UnsupportedRecruitmentTypeError,
    normalize_recruitment_type,
    recruitment_type_storage_aliases,
)


class RecruitmentTypeTest(unittest.TestCase):
    def test_normalizes_chinese_and_english_aliases(self) -> None:
        self.assertEqual("internship", normalize_recruitment_type("实习"))
        self.assertEqual(
            "internship",
            normalize_recruitment_type("日常实习项目"),
        )
        self.assertEqual("internship", normalize_recruitment_type("Intern"))
        self.assertEqual("campus", normalize_recruitment_type("校园招聘"))
        self.assertEqual("experienced", normalize_recruitment_type("社招"))

    def test_rejects_unknown_value(self) -> None:
        with self.assertRaises(UnsupportedRecruitmentTypeError):
            normalize_recruitment_type("兼职")

    def test_exposes_legacy_storage_aliases(self) -> None:
        aliases = recruitment_type_storage_aliases("internship")

        self.assertIn("internship", aliases)
        self.assertIn("实习", aliases)


if __name__ == "__main__":
    unittest.main()
