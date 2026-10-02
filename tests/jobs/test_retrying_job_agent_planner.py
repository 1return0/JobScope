from __future__ import annotations

from unittest import TestCase

from app.application.jobs.job_agent_planning import AgentToolDefinition, JobAgentPlan
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    JobAgentPlannerAuthenticationError,
    JobAgentPlannerUnavailableError,
)
from app.infrastructure.llm.retrying_job_agent_planner import (
    JobAgentPlannerRetryPolicy,
    RetryingJobAgentPlanner,
)


class _Planner:
    identity = "test-planner"

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = iter(outcomes)
        self.calls = 0

    def plan(self, *, user_query: str, tools: tuple[AgentToolDefinition, ...]) -> JobAgentPlan:
        self.calls += 1
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


class RetryingJobAgentPlannerTest(TestCase):
    def test_retries_one_transient_unavailable_error_then_returns_plan(self) -> None:
        base = _Planner([
            JobAgentPlannerUnavailableError("temporary"),
            JobAgentPlan(decision="refuse", reason="safe refusal"),
        ])
        delays: list[float] = []
        planner = RetryingJobAgentPlanner(
            base,
            JobAgentPlannerRetryPolicy(max_attempts=2, initial_backoff_seconds=0.2),
            sleeper=delays.append,
        )

        result = planner.plan(user_query="question", tools=())

        self.assertEqual("refuse", result.decision)
        self.assertEqual(2, base.calls)
        self.assertEqual([0.2], delays)

    def test_does_not_retry_authentication_failure(self) -> None:
        base = _Planner([JobAgentPlannerAuthenticationError("invalid key")])
        planner = RetryingJobAgentPlanner(base, sleeper=lambda _: None)

        with self.assertRaises(JobAgentPlannerAuthenticationError):
            planner.plan(user_query="question", tools=())

        self.assertEqual(1, base.calls)
