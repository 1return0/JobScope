from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.application.jobs.job_tool_graph import AgentToolCall


@dataclass(frozen=True, slots=True)
class AgentToolDefinition:
    name: str
    description: str
    arguments_schema: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent tool name must not be blank")
        if not self.description.strip():
            raise ValueError("Agent tool description must not be blank")
        if not self.arguments_schema:
            raise ValueError("Agent tool arguments schema must not be empty")


class JobAgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["call_tool", "refuse"]
    reason: str = Field(min_length=1, max_length=1000)
    tool_call: AgentToolCall | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> JobAgentPlan:
        if self.decision == "call_tool" and self.tool_call is None:
            raise ValueError("call_tool decision requires tool_call")
        if self.decision == "refuse" and self.tool_call is not None:
            raise ValueError("refuse decision must not contain tool_call")
        return self


@dataclass(frozen=True, slots=True)
class JobAgentPlannerTokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.total_tokens) < 0:
            raise ValueError("planner token usage must not be negative")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("planner total tokens must equal input plus output")


@dataclass(frozen=True, slots=True)
class ObservedJobAgentPlan:
    plan: JobAgentPlan
    token_usage: JobAgentPlannerTokenUsage | None


class JobAgentPlanner(Protocol):
    @property
    def identity(self) -> str:
        ...

    def plan(
        self,
        *,
        user_query: str,
        tools: tuple[AgentToolDefinition, ...],
    ) -> JobAgentPlan:
        ...


@runtime_checkable
class ObservableJobAgentPlanner(Protocol):
    def plan_observed(
        self,
        *,
        user_query: str,
        tools: tuple[AgentToolDefinition, ...],
    ) -> ObservedJobAgentPlan:
        ...
