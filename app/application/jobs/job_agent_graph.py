from __future__ import annotations

from typing import Any, Iterable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.application.jobs.job_agent_planning import (
    AgentToolDefinition,
    JobAgentPlanner,
)
from app.application.jobs.job_tool_graph import (
    AgentCallableTool,
    JobToolExecutionNode,
)


class JobAgentState(TypedDict, total=False):
    user_query: str
    planning_status: Literal["tool_selected", "refused"]
    planner_identity: str
    plan_reason: str
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    tool_error: dict[str, Any] | None


class JobAgentPlanningNode:
    def __init__(
        self,
        planner: JobAgentPlanner,
        tools: tuple[AgentCallableTool, ...],
    ) -> None:
        if not planner.identity.strip():
            raise ValueError("Job Agent planner identity must not be blank")
        self._planner = planner
        self._tool_definitions = tuple(
            AgentToolDefinition(
                name=tool.name,
                description=tool.description,
                arguments_schema=tool.arguments_schema,
            )
            for tool in tools
        )

    def __call__(self, state: JobAgentState) -> JobAgentState:
        user_query = (state.get("user_query") or "").strip()
        if not user_query:
            raise ValueError("Job Agent user_query must not be blank")

        plan = self._planner.plan(
            user_query=user_query,
            tools=self._tool_definitions,
        )
        if plan.decision == "refuse":
            return {
                "planning_status": "refused",
                "planner_identity": self._planner.identity,
                "plan_reason": plan.reason,
                "tool_call": None,
                "tool_result": None,
                "tool_error": None,
            }
        if plan.tool_call is None:
            raise AssertionError("validated call_tool plan requires tool_call")
        return {
            "planning_status": "tool_selected",
            "planner_identity": self._planner.identity,
            "plan_reason": plan.reason,
            "tool_call": plan.tool_call.model_dump(mode="json"),
            "tool_result": None,
            "tool_error": None,
        }


def _route_after_planning(
    state: JobAgentState,
) -> Literal["execute_tool", "end"]:
    if state.get("planning_status") == "tool_selected":
        return "execute_tool"
    return "end"


def build_job_agent_graph(
    planner: JobAgentPlanner,
    tools: Iterable[AgentCallableTool],
):
    registered_tools = tuple(tools)
    if not registered_tools:
        raise ValueError("at least one Agent tool is required")

    builder = StateGraph(JobAgentState)
    builder.add_node(
        "plan",
        JobAgentPlanningNode(planner, registered_tools),
    )
    builder.add_node(
        "execute_tool",
        JobToolExecutionNode(registered_tools),
    )
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        _route_after_planning,
        {
            "execute_tool": "execute_tool",
            "end": END,
        },
    )
    builder.add_edge("execute_tool", END)
    return builder.compile()
