"""Bounded retry for idempotent planner model requests only."""

from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Callable

from app.application.jobs.job_agent_planning import (
    AgentToolDefinition,
    JobAgentPlan,
    JobAgentPlanner,
)
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    JobAgentPlannerRateLimitError,
    JobAgentPlannerUnavailableError,
)


@dataclass(frozen=True, slots=True)
class JobAgentPlannerRetryPolicy:
    max_attempts: int = 2
    initial_backoff_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("planner max_attempts must be at least one")
        if self.initial_backoff_seconds < 0:
            raise ValueError("planner backoff must not be negative")

    def delay_before_retry(self, completed_attempts: int) -> float:
        return self.initial_backoff_seconds * (2 ** (completed_attempts - 1))


class RetryingJobAgentPlanner:
    def __init__(
        self,
        delegate: JobAgentPlanner,
        policy: JobAgentPlannerRetryPolicy | None = None,
        *,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self._delegate = delegate
        self._policy = policy or JobAgentPlannerRetryPolicy()
        self._sleeper = sleeper

    @property
    def identity(self) -> str:
        return f"retry({self._delegate.identity};max_attempts={self._policy.max_attempts})"

    def plan(
        self, *, user_query: str, tools: tuple[AgentToolDefinition, ...]
    ) -> JobAgentPlan:
        for attempt in range(1, self._policy.max_attempts + 1):
            try:
                return self._delegate.plan(user_query=user_query, tools=tools)
            except (JobAgentPlannerUnavailableError, JobAgentPlannerRateLimitError):
                if attempt == self._policy.max_attempts:
                    raise
                self._sleeper(self._policy.delay_before_retry(attempt))
        raise AssertionError("planner retry loop must return or raise")

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()
