import unittest
from datetime import date

from pydantic import ValidationError

from app.application.jobs.current_job_records import (
    CurrentJobRecordFilters,
    CurrentStructuredJobRecord,
)
from app.application.jobs.job_search_tool import (
    SearchCurrentJobsArguments,
    SearchCurrentJobsTool,
)


class _Reader:
    def __init__(
        self,
        rows: list[CurrentStructuredJobRecord],
    ) -> None:
        self._rows = rows
        self.received_filters: CurrentJobRecordFilters | None = None

    def list_current(
        self,
        filters: CurrentJobRecordFilters | None = None,
    ) -> list[CurrentStructuredJobRecord]:
        self.received_filters = filters
        return self._rows


def _record() -> CurrentStructuredJobRecord:
    return CurrentStructuredJobRecord(
        record_id="job_" + "1" * 64,
        source_snapshot_id="snap_" + "2" * 64,
        company="示例科技",
        job_title="AI Agent实习生",
        locations=("上海", "北京"),
        education_requirement="本科及以上",
        major_requirement="专业不限",
        recruitment_type="校园招聘",
        application_deadline_raw="2026年9月30日",
        application_deadline_normalized=date(2026, 9, 30),
        deadline_normalization_status="normalized",
    )


class SearchCurrentJobsToolTest(unittest.TestCase):
    def test_exposes_stable_name_description_and_json_schema(self) -> None:
        tool = SearchCurrentJobsTool(_Reader([]))

        schema = tool.arguments_schema

        self.assertEqual("search_current_jobs", tool.name)
        self.assertIn("current", tool.description.casefold())
        self.assertIn("job_title", schema["properties"])
        self.assertIn("location", schema["properties"])
        self.assertIn("recruitment_type", schema["properties"])
        self.assertFalse(schema["additionalProperties"])

    def test_converts_job_title_and_recruitment_type_filters(self) -> None:
        reader = _Reader([_record()])
        tool = SearchCurrentJobsTool(reader)

        output = tool.execute(
            SearchCurrentJobsArguments(
                job_title="AI Agent",
                recruitment_type="campus",
            )
        )

        self.assertEqual("found", output.status)
        self.assertEqual("AI Agent", reader.received_filters.job_title)
        self.assertEqual("campus", reader.received_filters.recruitment_type)

    def test_normalizes_chinese_recruitment_type_before_querying(self) -> None:
        reader = _Reader([])
        tool = SearchCurrentJobsTool(reader)

        tool.invoke({"recruitment_type": "实习"})

        self.assertEqual(
            "internship",
            reader.received_filters.recruitment_type,
        )

    def test_converts_validated_arguments_to_business_filters(self) -> None:
        reader = _Reader([_record()])
        tool = SearchCurrentJobsTool(reader)

        output = tool.execute(
            SearchCurrentJobsArguments(
                company="示例科技",
                location="上海",
                deadline_on_or_after=date(2026, 9, 1),
                deadline_on_or_before=date(2026, 9, 30),
                limit=5,
            )
        )

        self.assertEqual("found", output.status)
        self.assertEqual(1, output.result_count)
        self.assertEqual(("上海", "北京"), output.jobs[0].locations)
        self.assertEqual("上海", reader.received_filters.location)
        self.assertEqual(5, reader.received_filters.limit)

    def test_returns_explicit_no_matches_instead_of_inventing_jobs(self) -> None:
        output = SearchCurrentJobsTool(_Reader([])).execute(
            SearchCurrentJobsArguments(location="广州")
        )

        self.assertEqual("no_matches", output.status)
        self.assertEqual(0, output.result_count)
        self.assertEqual((), output.jobs)

    def test_rejects_invalid_model_arguments_before_querying(self) -> None:
        reader = _Reader([])

        with self.assertRaises(ValidationError):
            SearchCurrentJobsArguments(
                deadline_on_or_after=date(2026, 10, 1),
                deadline_on_or_before=date(2026, 9, 1),
            )

        self.assertIsNone(reader.received_filters)


if __name__ == "__main__":
    unittest.main()
