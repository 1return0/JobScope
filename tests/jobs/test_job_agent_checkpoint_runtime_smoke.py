from __future__ import annotations

import os
import unittest

from app.application.jobs.job_agent_planning import JobAgentPlan
from app.application.jobs.job_agent_thread_identity import (
    build_job_agent_thread_id,
)
from app.application.jobs.job_tool_graph import AgentToolCall
from app.config import load_settings
from app.runtime_composition import compose_job_agent_runtime


class _Planner:
    identity = "job-agent-checkpoint-smoke-planner"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def plan(self, *, user_query, tools):
        self.queries.append(user_query)
        return JobAgentPlan(
            decision="call_tool",
            reason="current jobs are needed",
            tool_call=AgentToolCall(
                call_id="checkpoint-smoke-call",
                name="search_current_jobs",
                arguments={"limit": 1},
            ),
        )


@unittest.skipUnless(
    os.getenv("JOBSCOPE_RUN_POSTGRES_JOB_AGENT_MEMORY_SMOKE") == "true",
    "set JOBSCOPE_RUN_POSTGRES_JOB_AGENT_MEMORY_SMOKE=true to run PostgreSQL smoke",
)
class JobAgentCheckpointRuntimeSmokeTest(unittest.TestCase):
    def test_runtime_restores_same_owner_query_context(self) -> None:
        owner_id = "job-agent-memory-smoke-owner"
        planner = _Planner()
        runtime = compose_job_agent_runtime(load_settings(), planner=planner)
        try:
            first = runtime.invoke("找上海岗位", owner_id=owner_id)
            second = runtime.invoke("只看校招", owner_id=owner_id)

            self.assertEqual(
                [
                    {"role": "user", "content": "找上海岗位"},
                    {
                        "role": "assistant",
                        "content": "found: registered read-only tool execution was completed",
                    },
                ],
                first["conversation_turns"],
            )
            self.assertEqual(
                [
                    {"role": "user", "content": "找上海岗位"},
                    {
                        "role": "assistant",
                        "content": "found: registered read-only tool execution was completed",
                    },
                    {"role": "user", "content": "只看校招"},
                    {
                        "role": "assistant",
                        "content": "found: registered read-only tool execution was completed",
                    },
                ],
                second["conversation_turns"],
            )
            self.assertIn("找上海岗位", planner.queries[1])
            self.assertIn("只看校招", planner.queries[1])
        finally:
            runtime.graph.checkpointer.delete_thread(
                build_job_agent_thread_id(owner_id)
            )
            runtime.close()
