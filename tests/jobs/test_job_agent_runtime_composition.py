import unittest

from langgraph.checkpoint.memory import MemorySaver
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
        self.queries = []

    def plan(self, *, user_query, tools):
        self.queries.append(user_query)
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


class _MemoryService:
    def list_preferences(self, *, owner_id: str):
        from datetime import datetime, timezone
        from app.application.memory.long_term_memory import LongTermMemoryEntry

        return (
            LongTermMemoryEntry(
                owner_id=owner_id,
                preference_key="preferred_location",
                preference_value="Shanghai",
                consented_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                revision=1,
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
            checkpointing_enabled=False,
        )

        try:
            state = runtime.invoke(
                "Find current Shanghai jobs",
                owner_id="test-owner",
            )
        finally:
            runtime.close()

        self.assertEqual("tool_selected", state["planning_status"])
        self.assertEqual("search_current_jobs", state["tool_result"]["tool_name"])
        self.assertEqual("no_matches", state["tool_result"]["output"]["status"])
        self.assertEqual(0, state["tool_result"]["output"]["result_count"])
        self.assertIsNone(state["tool_error"])
        self.assertEqual(0, state["harness_snapshot"].model_call_count)
        self.assertEqual(1, state["harness_snapshot"].tool_call_count)

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
            checkpointing_enabled=False,
        )

        try:
            runtime.invoke(
                "Find current Shanghai jobs",
                owner_id="test-owner",
            )
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

    def test_runtime_loads_preferences_as_non_checkpointed_graph_context(self) -> None:
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
            memory_service=_MemoryService(),
            checkpointing_enabled=False,
        )

        try:
            runtime.invoke("Find internships", owner_id="test-owner")
        finally:
            runtime.close()

        self.assertIn("preferred_location: Shanghai", planner.queries[0])
        self.assertIn("Current user request:\nFind internships", planner.queries[0])

    def test_runtime_reads_and_clears_only_its_derived_checkpoint_thread(self) -> None:
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
            checkpointer=MemorySaver(),
        )

        try:
            runtime.invoke("Find Shanghai jobs", owner_id="owner-a")
            self.assertEqual(
                (
                    {"role": "user", "content": "Find Shanghai jobs"},
                    {
                        "role": "assistant",
                        "content": "no_matches: registered read-only tool execution was completed",
                    },
                ),
                runtime.get_conversation_turns(owner_id="owner-a"),
            )
            self.assertTrue(runtime.clear_conversation_turns(owner_id="owner-a"))
            self.assertEqual((), runtime.get_conversation_turns(owner_id="owner-a"))
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
