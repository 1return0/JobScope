from __future__ import annotations

from typing import Any, Iterable, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.application.jobs.agent_harness import (
    JobAgentHarness,
    JobAgentHarnessLimitExceededError,
    bind_job_agent_harness,
    invoke_harnessed_tool,
)


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any]


class AgentCallableTool(Protocol):
    name: str
    description: str

    @property
    def arguments_schema(self) -> dict[str, Any]:
        ...

    def invoke(self, raw_arguments: dict[str, Any]) -> BaseModel:
        ...


class JobToolGraphState(TypedDict, total=False):
    tool_call: dict[str, Any]
    tool_result: dict[str, Any] | None
    tool_error: dict[str, Any] | None


class JobToolExecutionNode:
    def __init__(self, tools: Iterable[AgentCallableTool]) -> None:
        registry: dict[str, AgentCallableTool] = {}
        for tool in tools:
            if tool.name in registry:
                raise ValueError(f"duplicate Agent tool name: {tool.name}")
            registry[tool.name] = tool
        if not registry:
            raise ValueError("at least one Agent tool is required")
        self._registry = registry

    def __call__(
        self,
        state: JobToolGraphState,
        runtime: Runtime[Any] | None = None,
    ) -> JobToolGraphState:
        try:
            tool_call = AgentToolCall.model_validate(
                state.get("tool_call")
            )
        except ValidationError:
            return {
                "tool_result": None,
                "tool_error": {
                    "code": "invalid-tool-call",
                    "message": "tool call does not match the required contract",
                },
            }

        tool = self._registry.get(tool_call.name)
        if tool is None:
            return {
                "tool_result": None,
                "tool_error": {
                    "call_id": tool_call.call_id,
                    "code": "unknown-tool",
                    "message": f"tool is not registered: {tool_call.name}",
                },
            }

        try:
            context = runtime.context if runtime is not None else {}
            harness = context.get("harness") if context else None
            if isinstance(harness, JobAgentHarness):
                with bind_job_agent_harness(harness):
                    output = invoke_harnessed_tool(
                        tool_call.name,
                        lambda: tool.invoke(tool_call.arguments),
                    )
            else:
                output = tool.invoke(tool_call.arguments)
        except ValidationError:
            return {
                "tool_result": None,
                "tool_error": {
                    "call_id": tool_call.call_id,
                    "code": "invalid-tool-arguments",
                    "message": (
                        "tool arguments do not match the registered schema"
                    ),
                },
            }
        except JobAgentHarnessLimitExceededError:
            raise
        except RuntimeError:
            # Provider, capacity and prepared-service failures are operational
            # tool failures. Keep their internals out of the Agent response.
            return {
                "tool_result": None,
                "tool_error": {
                    "call_id": tool_call.call_id,
                    "code": "tool-execution-failed",
                    "message": "the selected tool is temporarily unavailable",
                },
            }

        return {
            "tool_result": {
                "call_id": tool_call.call_id,
                "tool_name": tool_call.name,
                "output": output.model_dump(mode="json"),
            },
            "tool_error": None,
        }


def build_job_tool_execution_graph(
    tools: Iterable[AgentCallableTool],
):
    builder = StateGraph(JobToolGraphState)
    builder.add_node("execute_tool", JobToolExecutionNode(tools))
    builder.add_edge(START, "execute_tool")
    builder.add_edge("execute_tool", END)
    return builder.compile()
