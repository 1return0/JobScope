from __future__ import annotations

from unittest import TestCase

from app.application.jobs.agent_harness import (
    HarnessedJobAgentPlanner,
    JobAgentHarness,
    JobAgentHarnessLimitExceededError,
    JobAgentHarnessLimits,
    bind_job_agent_harness,
)
from app.application.jobs.job_agent_planning import AgentToolDefinition, JobAgentPlan


class _Planner:
    identity = "test-model-planner"

    def __init__(self) -> None:
        self.calls = 0

    def plan(self, *, user_query: str, tools: tuple[AgentToolDefinition, ...]) -> JobAgentPlan:
        self.calls += 1
        return JobAgentPlan(decision="refuse", reason="test refusal")


class JobAgentHarnessTest(TestCase):
    def test_model_wrapper_counts_only_when_request_harness_is_bound(self) -> None:
        delegate = _Planner()
        planner = HarnessedJobAgentPlanner(delegate)
        harness = JobAgentHarness()

        planner.plan(user_query="one", tools=())
        with bind_job_agent_harness(harness):
            planner.plan(user_query="two", tools=())

        self.assertEqual(2, delegate.calls)
        self.assertEqual(1, harness.snapshot().model_call_count)
        self.assertEqual("job_agent_planner_model", harness.snapshot().steps[0].operation_name)

    def test_rejects_model_call_before_executing_it_at_the_fixed_limit(self) -> None:
        delegate = _Planner()
        planner = HarnessedJobAgentPlanner(delegate)
        harness = JobAgentHarness(JobAgentHarnessLimits(max_model_calls=1))

        with bind_job_agent_harness(harness):
            planner.plan(user_query="one", tools=())
            with self.assertRaisesRegex(
                JobAgentHarnessLimitExceededError,
                "job-agent-harness-model-call-limit-reached",
            ):
                planner.plan(user_query="two", tools=())

        self.assertEqual(1, delegate.calls)
        self.assertEqual("rejected", harness.snapshot().steps[-1].status)
