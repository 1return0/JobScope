import unittest
from datetime import date

from pydantic import ValidationError

from app.application.jobs.current_job_records import (
    CurrentJobRecordFilters,
    CurrentStructuredJobRecord,
)
from app.application.jobs.job_agent_graph import build_job_agent_graph
from app.application.jobs.job_agent_planning import JobAgentPlan
from app.application.jobs.job_search_tool import SearchCurrentJobsTool
from app.application.jobs.job_tool_graph import AgentToolCall


class _Reader:
    def __init__(self) -> None:
        self.calls = 0

    def list_current(
        self,
        filters: CurrentJobRecordFilters | None = None,
    ) -> list[CurrentStructuredJobRecord]:
        self.calls += 1
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


class _Planner:
    identity = "fake-job-agent-planner-v1"

    def __init__(self, plan: JobAgentPlan) -> None:
        self._plan = plan
        self.calls = 0
        self.received_query: str | None = None
        self.received_tools = ()

    def plan(self, *, user_query, tools):
        self.calls += 1
        self.received_query = user_query
        self.received_tools = tools
        return self._plan


def _tool_plan() -> JobAgentPlan:
    return JobAgentPlan(
        decision="call_tool",
        reason="The question asks for current jobs filtered by location.",
        tool_call=AgentToolCall(
            call_id="call-001",
            name="search_current_jobs",
            arguments={"location": "上海", "limit": 5},
        ),
    )


class JobAgentGraphTest(unittest.TestCase):
    def test_tool_plan_routes_through_execution_node(self) -> None:
        reader = _Reader()
        planner = _Planner(_tool_plan())
        graph = build_job_agent_graph(
            planner,
            [SearchCurrentJobsTool(reader)],
        )

        state = graph.invoke({"user_query": "帮我找上海的实习岗位"})

        self.assertEqual("tool_selected", state["planning_status"])
        self.assertEqual(planner.identity, state["planner_identity"])
        self.assertEqual("found", state["tool_result"]["output"]["status"])
        self.assertIsNone(state["tool_error"])
        self.assertEqual(1, reader.calls)
        self.assertEqual("帮我找上海的实习岗位", planner.received_query)
        self.assertEqual(
            "search_current_jobs",
            planner.received_tools[0].name,
        )
        self.assertIn(
            "location",
            planner.received_tools[0].arguments_schema["properties"],
        )

    def test_refusal_plan_routes_directly_to_end(self) -> None:
        reader = _Reader()
        planner = _Planner(
            JobAgentPlan(
                decision="refuse",
                reason="The request is unrelated to campus recruitment.",
            )
        )
        graph = build_job_agent_graph(
            planner,
            [SearchCurrentJobsTool(reader)],
        )

        state = graph.invoke({"user_query": "帮我预测明天的彩票号码"})

        self.assertEqual("refused", state["planning_status"])
        self.assertIsNone(state["tool_call"])
        self.assertIsNone(state["tool_result"])
        self.assertEqual(0, reader.calls)

    def test_plan_contract_rejects_inconsistent_decisions(self) -> None:
        with self.assertRaises(ValidationError):
            JobAgentPlan(
                decision="call_tool",
                reason="A tool is needed.",
                tool_call=None,
            )
        with self.assertRaises(ValidationError):
            JobAgentPlan(
                decision="refuse",
                reason="No tool should run.",
                tool_call=_tool_plan().tool_call,
            )

    def test_blank_query_stops_before_planner_and_tool(self) -> None:
        reader = _Reader()
        planner = _Planner(_tool_plan())
        graph = build_job_agent_graph(
            planner,
            [SearchCurrentJobsTool(reader)],
        )

        with self.assertRaisesRegex(ValueError, "user_query"):
            graph.invoke({"user_query": "   "})

        self.assertEqual(0, planner.calls)
        self.assertEqual(0, reader.calls)


if __name__ == "__main__":
    unittest.main()
