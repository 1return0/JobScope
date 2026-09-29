import unittest
from datetime import date
from typing import Any

from pydantic import BaseModel

from app.application.jobs.current_job_records import (
    CurrentJobRecordFilters,
    CurrentStructuredJobRecord,
)
from app.application.jobs.job_search_tool import SearchCurrentJobsTool
from app.application.jobs.job_tool_graph import (
    JobToolExecutionNode,
    build_job_tool_execution_graph,
)


class _Reader:
    def __init__(self) -> None:
        self.calls = 0
        self.received_filters: CurrentJobRecordFilters | None = None

    def list_current(
        self,
        filters: CurrentJobRecordFilters | None = None,
    ) -> list[CurrentStructuredJobRecord]:
        self.calls += 1
        self.received_filters = filters
        return [
            CurrentStructuredJobRecord(
                record_id="job_" + "1" * 64,
                source_snapshot_id="snap_" + "2" * 64,
                company="示例科技",
                job_title="AI Agent实习生",
                locations=("上海",),
                education_requirement="本科及以上",
                major_requirement="专业不限",
                recruitment_type="校园招聘",
                application_deadline_raw="2026年9月30日",
                application_deadline_normalized=date(2026, 9, 30),
                deadline_normalization_status="normalized",
            )
        ]


class _EmptyArguments(BaseModel):
    pass


class _UnavailableTool:
    name = "unavailable_tool"
    description = "Always unavailable for a boundary test."

    @property
    def arguments_schema(self) -> dict[str, Any]:
        return _EmptyArguments.model_json_schema()

    def invoke(self, raw_arguments: dict[str, Any]) -> BaseModel:
        raise RuntimeError("provider secret detail")


class JobToolExecutionGraphTest(unittest.TestCase):
    def test_graph_dispatches_valid_call_and_keeps_call_identity(self) -> None:
        reader = _Reader()
        graph = build_job_tool_execution_graph(
            [SearchCurrentJobsTool(reader)]
        )

        state = graph.invoke(
            {
                "tool_call": {
                    "call_id": "call-001",
                    "name": "search_current_jobs",
                    "arguments": {
                        "location": "上海",
                        "limit": 5,
                    },
                }
            }
        )

        self.assertIsNone(state["tool_error"])
        self.assertEqual("call-001", state["tool_result"]["call_id"])
        self.assertEqual(
            "search_current_jobs",
            state["tool_result"]["tool_name"],
        )
        self.assertEqual("found", state["tool_result"]["output"]["status"])
        self.assertEqual(1, reader.calls)
        self.assertEqual("上海", reader.received_filters.location)

    def test_invalid_arguments_stop_before_reader_call(self) -> None:
        reader = _Reader()
        graph = build_job_tool_execution_graph(
            [SearchCurrentJobsTool(reader)]
        )

        state = graph.invoke(
            {
                "tool_call": {
                    "call_id": "call-002",
                    "name": "search_current_jobs",
                    "arguments": {"limit": 1000},
                }
            }
        )

        self.assertIsNone(state["tool_result"])
        self.assertEqual(
            "invalid-tool-arguments",
            state["tool_error"]["code"],
        )
        self.assertEqual(0, reader.calls)

    def test_unknown_tool_is_not_executed(self) -> None:
        reader = _Reader()
        graph = build_job_tool_execution_graph(
            [SearchCurrentJobsTool(reader)]
        )

        state = graph.invoke(
            {
                "tool_call": {
                    "call_id": "call-003",
                    "name": "delete_all_jobs",
                    "arguments": {},
                }
            }
        )

        self.assertEqual("unknown-tool", state["tool_error"]["code"])
        self.assertEqual(0, reader.calls)

    def test_duplicate_tool_names_fail_during_graph_construction(self) -> None:
        reader = _Reader()
        tool = SearchCurrentJobsTool(reader)

        with self.assertRaisesRegex(ValueError, "duplicate Agent tool name"):
            JobToolExecutionNode([tool, tool])

    def test_operational_tool_failure_returns_sanitized_error(self) -> None:
        graph = build_job_tool_execution_graph([_UnavailableTool()])

        state = graph.invoke(
            {
                "tool_call": {
                    "call_id": "call-004",
                    "name": "unavailable_tool",
                    "arguments": {},
                }
            }
        )

        self.assertIsNone(state["tool_result"])
        self.assertEqual(
            "tool-execution-failed",
            state["tool_error"]["code"],
        )
        self.assertNotIn("secret", state["tool_error"]["message"])


if __name__ == "__main__":
    unittest.main()
