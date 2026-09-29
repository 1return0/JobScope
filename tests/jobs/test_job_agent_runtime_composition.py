import unittest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.application.jobs.job_agent_planning import JobAgentPlan
from app.application.jobs.job_tool_graph import AgentToolCall
from app.config import Settings
from app.infrastructure.persistence.database import Base
from app.runtime_composition import compose_job_agent_runtime


class _Planner:
    identity = "fake-runtime-planner-v1"

    def __init__(self) -> None:
        self.tool_names = ()

    def plan(self, *, user_query, tools):
        self.tool_names = tuple(tool.name for tool in tools)
        return JobAgentPlan(
            decision="call_tool",
            reason="Current job records are required.",
            tool_call=AgentToolCall(
                call_id="runtime-call-1",
                name="search_current_jobs",
                arguments={"location": "Shanghai", "limit": 5},
            ),
        )


class JobAgentRuntimeCompositionTest(unittest.TestCase):
    def test_composes_graph_tool_and_sqlalchemy_query(self) -> None:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        runtime = compose_job_agent_runtime(
            Settings(),
            planner=_Planner(),
            engine=engine,
        )

        try:
            state = runtime.invoke("Find current Shanghai jobs")
        finally:
            runtime.close()

        self.assertEqual("tool_selected", state["planning_status"])
        self.assertEqual("search_current_jobs", state["tool_result"]["tool_name"])
        self.assertEqual("no_matches", state["tool_result"]["output"]["status"])
        self.assertEqual(0, state["tool_result"]["output"]["result_count"])
        self.assertIsNone(state["tool_error"])

    def test_registers_rag_and_mixed_tools_when_answer_service_exists(self) -> None:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        planner = _Planner()
        runtime = compose_job_agent_runtime(
            Settings(),
            planner=planner,
            engine=engine,
            answer_service=object(),
        )

        try:
            runtime.invoke("Find current Shanghai jobs")
        finally:
            runtime.close()

        self.assertEqual(
            (
                "search_current_jobs",
                "answer_recruitment_question",
                "search_jobs_with_evidence",
            ),
            planner.tool_names,
        )


if __name__ == "__main__":
    unittest.main()
