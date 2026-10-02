"""Request-scoped execution limits for the Job Agent."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Generator, Literal, Protocol, TypeVar


T = TypeVar("T")
OperationKind = Literal["model", "tool"]
StepStatus = Literal["succeeded", "failed", "rejected"]
_ACTIVE_HARNESS: ContextVar["JobAgentHarness | None"] = ContextVar(
    "job_agent_active_harness", default=None
)


class JobAgentHarnessLimitExceededError(RuntimeError):
    def __init__(self, failure_code: str) -> None:
        super().__init__(failure_code)
        self.failure_code = failure_code


class _PlannerDelegate(Protocol):
    @property
    def identity(self) -> str:
        ...

    def plan(self, *, user_query: str, tools: tuple[Any, ...]) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class JobAgentHarnessLimits:
    max_model_calls: int = 3
    max_tool_calls: int = 4
    max_elapsed_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_model_calls < 0 or self.max_tool_calls < 0:
            raise ValueError("Job Agent harness call limits must not be negative")
        if self.max_elapsed_seconds <= 0:
            raise ValueError("Job Agent harness elapsed limit must be positive")


@dataclass(frozen=True, slots=True)
class JobAgentHarnessStep:
    operation_kind: OperationKind
    operation_name: str
    status: StepStatus
    elapsed_seconds: float
    failure_code: str | None = None


@dataclass(frozen=True, slots=True)
class JobAgentHarnessSnapshot:
    model_call_count: int
    tool_call_count: int
    elapsed_seconds: float
    steps: tuple[JobAgentHarnessStep, ...]


class JobAgentHarness:
    def __init__(
        self,
        limits: JobAgentHarnessLimits | None = None,
        *,
        time_source: Callable[[], float] = perf_counter,
    ) -> None:
        self._limits = limits or JobAgentHarnessLimits()
        self._time_source = time_source
        self._started_at = time_source()
        self._model_call_count = 0
        self._tool_call_count = 0
        self._steps: list[JobAgentHarnessStep] = []

    def invoke_model(self, operation_name: str, operation: Callable[[], T]) -> T:
        return self._invoke("model", operation_name, operation)

    def invoke_tool(self, operation_name: str, operation: Callable[[], T]) -> T:
        return self._invoke("tool", operation_name, operation)

    def snapshot(self) -> JobAgentHarnessSnapshot:
        return JobAgentHarnessSnapshot(
            model_call_count=self._model_call_count,
            tool_call_count=self._tool_call_count,
            elapsed_seconds=self._elapsed_seconds(),
            steps=tuple(self._steps),
        )

    def _invoke(
        self, kind: OperationKind, operation_name: str, operation: Callable[[], T]
    ) -> T:
        if not operation_name.strip():
            raise ValueError("Job Agent harness operation_name must not be blank")
        failure_code = self._limit_failure(kind)
        if failure_code is not None:
            self._steps.append(JobAgentHarnessStep(
                kind, operation_name, "rejected", 0.0, failure_code
            ))
            raise JobAgentHarnessLimitExceededError(failure_code)
        if kind == "model":
            self._model_call_count += 1
        else:
            self._tool_call_count += 1
        started_at = self._time_source()
        try:
            result = operation()
        except Exception as error:
            self._steps.append(JobAgentHarnessStep(
                kind, operation_name, "failed", self._time_source() - started_at,
                _safe_failure_code(error),
            ))
            raise
        self._steps.append(JobAgentHarnessStep(
            kind, operation_name, "succeeded", self._time_source() - started_at
        ))
        return result

    def _limit_failure(self, kind: OperationKind) -> str | None:
        if self._elapsed_seconds() >= self._limits.max_elapsed_seconds:
            return "job-agent-harness-elapsed-time-limit-reached"
        if kind == "model" and self._model_call_count >= self._limits.max_model_calls:
            return "job-agent-harness-model-call-limit-reached"
        if kind == "tool" and self._tool_call_count >= self._limits.max_tool_calls:
            return "job-agent-harness-tool-call-limit-reached"
        return None

    def _elapsed_seconds(self) -> float:
        return self._time_source() - self._started_at


@contextmanager
def bind_job_agent_harness(harness: JobAgentHarness) -> Generator[None, None, None]:
    token: Token[JobAgentHarness | None] = _ACTIVE_HARNESS.set(harness)
    try:
        yield
    finally:
        _ACTIVE_HARNESS.reset(token)


class HarnessedJobAgentPlanner:
    """Counts only a model-backed planner that is explicitly wrapped at compose time."""

    def __init__(self, delegate: _PlannerDelegate) -> None:
        self._delegate = delegate

    @property
    def identity(self) -> str:
        return self._delegate.identity

    def plan(
        self, *, user_query: str, tools: tuple[Any, ...]
    ) -> Any:
        active = _ACTIVE_HARNESS.get()
        operation = lambda: self._delegate.plan(user_query=user_query, tools=tools)
        if active is None:
            return operation()
        return active.invoke_model("job_agent_planner_model", operation)

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()


def invoke_harnessed_tool(operation_name: str, operation: Callable[[], T]) -> T:
    active = _ACTIVE_HARNESS.get()
    if active is None:
        return operation()
    return active.invoke_tool(operation_name, operation)


def _safe_failure_code(error: Exception) -> str:
    code = getattr(error, "failure_code", None)
    if isinstance(code, str) and code.strip():
        return code
    return "job-agent-harness-operation-failed"
