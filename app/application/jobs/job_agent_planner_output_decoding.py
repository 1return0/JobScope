from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.application.jobs.job_agent_planning import JobAgentPlan
from app.application.jobs.job_tool_graph import AgentToolCall


class JobAgentPlanningOutputDecodingError(ValueError):
    pass


class _PlannerModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["call_tool", "refuse"]
    reason: str = Field(min_length=1, max_length=1000)
    tool_name: str | None
    arguments: dict[str, Any] | None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> _PlannerModelOutput:
        if self.decision == "call_tool":
            if self.tool_name is None or self.arguments is None:
                raise ValueError(
                    "call_tool requires tool_name and arguments"
                )
        elif self.tool_name is not None or self.arguments is not None:
            raise ValueError(
                "refuse requires null tool_name and arguments"
            )
        return self


class JobAgentPlanningOutputDecoder:
    def decode(
        self,
        raw_output: str,
        *,
        call_id: str,
        allowed_tool_names: tuple[str, ...],
    ) -> JobAgentPlan:
        normalized_call_id = call_id.strip()
        if not normalized_call_id:
            raise ValueError("call_id must not be blank")
        if not allowed_tool_names:
            raise ValueError("allowed_tool_names must not be empty")
        try:
            payload = json.loads(raw_output)
            output = _PlannerModelOutput.model_validate(payload)
            if output.decision == "refuse":
                return JobAgentPlan(
                    decision="refuse",
                    reason=output.reason,
                )
            if output.tool_name not in set(allowed_tool_names):
                raise ValueError("planner selected an unregistered tool")
            if output.arguments is None or output.tool_name is None:
                raise AssertionError(
                    "validated call_tool output requires tool data"
                )
            return JobAgentPlan(
                decision="call_tool",
                reason=output.reason,
                tool_call=AgentToolCall(
                    call_id=normalized_call_id,
                    name=output.tool_name,
                    arguments=output.arguments,
                ),
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ValidationError,
            ValueError,
        ) as error:
            raise JobAgentPlanningOutputDecodingError(
                "model returned an invalid Job Agent plan"
            ) from error
