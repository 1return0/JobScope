from __future__ import annotations

from typing import Annotated, Any, Iterable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from app.application.jobs.job_agent_planning import (
    AgentToolDefinition,
    JobAgentPlanner,
)
from app.application.jobs.job_tool_graph import (
    AgentCallableTool,
    JobToolExecutionNode,
)
from app.application.memory.safe_conversation import (
    merge_safe_conversation_turns,
    safe_agent_summary,
    safe_conversation_turn,
)
from app.application.jobs.agent_harness import (
    JobAgentHarness,
    bind_job_agent_harness,
)


def _merge_conversation_turns(
    existing: list[dict[str, str] | str],
    new_values: list[dict[str, str] | str],
) -> list[dict[str, str]]:
    return merge_safe_conversation_turns(existing, new_values)


class JobAgentState(TypedDict, total=False):
    user_query: str
    # Added only by the trusted API/runtime boundary, never by the browser.
    owner_id: str
    conversation_turns: Annotated[list[dict[str, str]], _merge_conversation_turns]
    planning_status: Literal["tool_selected", "refused"]
    planner_identity: str
    plan_reason: str
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    tool_error: dict[str, Any] | None


class JobAgentInvocationContext(TypedDict, total=False):
    """Consented preferences for this invocation; never checkpointed."""

    preferences: dict[str, str]
    harness: JobAgentHarness


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

    def __call__(
        self,
        state: JobAgentState,
        runtime: Runtime[JobAgentInvocationContext],
    ) -> JobAgentState:
        user_query = (state.get("user_query") or "").strip()
        if not user_query:
            raise ValueError("Job Agent user_query must not be blank")

        history = tuple(
            merge_safe_conversation_turns(
                [],
                list(state.get("conversation_turns", ())),
            )
        )
        context = runtime.context or {}
        planner_query = _planner_query(
            history,
            user_query,
            preferences=context.get("preferences", {}),
        )
        harness = context.get("harness")
        if isinstance(harness, JobAgentHarness):
            with bind_job_agent_harness(harness):
                plan = self._planner.plan(
                    user_query=planner_query,
                    tools=self._tool_definitions,
                )
        else:
            plan = self._planner.plan(
                user_query=planner_query,
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
                "conversation_turns": [
                    safe_conversation_turn(
                        role="user", content=user_query
                    ).to_checkpoint()
                ],
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
            "conversation_turns": [
                safe_conversation_turn(
                    role="user", content=user_query
                ).to_checkpoint()
            ],
        }


class JobAgentConversationFinalizerNode:
    def __call__(self, state: JobAgentState) -> JobAgentState:
        outcome = state.get("planning_status", "unknown")
        reason = state.get("plan_reason", "no safe summary is available")
        if state.get("tool_error") is not None:
            outcome = "tool_error"
            reason = state["tool_error"].get("message", reason)
        elif state.get("tool_result") is not None:
            output = state["tool_result"].get("output", {})
            outcome = str(output.get("status", outcome))
            reason = "registered read-only tool execution was completed"
        return {"conversation_turns": [
            safe_agent_summary(outcome=outcome, reason=reason).to_checkpoint()
        ]}


def _route_after_planning(
    state: JobAgentState,
) -> Literal["execute_tool", "end"]:
    if state.get("planning_status") == "tool_selected":
        return "execute_tool"
    return "end"


def _planner_query(
    history: tuple[dict[str, str], ...],
    current_query: str,
    *,
    preferences: dict[str, str] | None = None,
) -> str:
    sections: list[str] = []
    if preferences:
        stable_preferences = "\n".join(
            f"- {key}: {value}"
            for key, value in sorted(preferences.items())
        )
        sections.append(
            "Consented user preferences (use only when relevant; "
            "the current request takes priority):\n"
            f"{stable_preferences}"
        )
    previous = "\n".join(
        f"- {turn['role']}: {turn['content']}" for turn in history
    )
    if previous:
        sections.append(
            "Previous user turns (context only; obey current request):\n"
            f"{previous}"
        )
    if not sections:
        return current_query
    sections.append(f"Current user request:\n{current_query}")
    return "\n\n".join(sections)


def build_job_agent_graph(
    planner: JobAgentPlanner,
    tools: Iterable[AgentCallableTool],
    *,
    checkpointer: Any | None = None,
):
    registered_tools = tuple(tools)
    if not registered_tools:
        raise ValueError("at least one Agent tool is required")

    builder = StateGraph(
        JobAgentState,
        context_schema=JobAgentInvocationContext,
    )
    builder.add_node(
        "plan",
        JobAgentPlanningNode(planner, registered_tools),
    )
    builder.add_node("record_assistant", JobAgentConversationFinalizerNode())
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
                "end": "record_assistant",
        },
    )
    builder.add_edge("execute_tool", "record_assistant")
    builder.add_edge("record_assistant", END)
    return builder.compile(checkpointer=checkpointer)
