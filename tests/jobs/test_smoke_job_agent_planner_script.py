import unittest

from app.application.jobs.job_agent_planning import JobAgentPlan
from app.application.jobs.job_tool_graph import AgentToolCall
from scripts.jobs.smoke_test_job_agent_planner import run_smoke


class _Planner:
    identity = "fake-planner-v1"

    def plan(self, *, user_query, tools):
        self.user_query = user_query
        self.tools = tools
        return JobAgentPlan(
            decision="call_tool",
            reason="Current JobScope data is required.",
            tool_call=AgentToolCall(
                call_id="call-001",
                name="search_current_jobs",
                arguments={"location": "上海"},
            ),
        )


class JobAgentPlannerSmokeScriptTest(unittest.TestCase):
    def test_report_contains_identity_plan_and_limit_scope(self) -> None:
        planner = _Planner()

        report = run_smoke(
            planner,
            user_query="帮我找上海的实习岗位",
        )

        self.assertEqual("passed", report["status"])
        self.assertEqual(planner.identity, report["planner_identity"])
        self.assertEqual("call_tool", report["plan"]["decision"])
        self.assertEqual(
            "search_current_jobs",
            planner.tools[0].name,
        )
        self.assertIn("no tool execution", report["limitations"])


if __name__ == "__main__":
    unittest.main()
