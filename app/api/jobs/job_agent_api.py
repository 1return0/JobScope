from __future__ import annotations

from typing import Any, Literal, Protocol

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.application.jobs.job_agent_planner_output_decoding import (
    JobAgentPlanningOutputDecodingError,
)
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    JobAgentPlannerAuthenticationError,
    JobAgentPlannerRateLimitError,
    JobAgentPlannerResponseError,
    JobAgentPlannerUnavailableError,
    JobAgentPlannerUpstreamError,
)


class JobAgentRuntimeProvider(Protocol):
    def invoke(self, user_query: str) -> dict[str, Any]:
        ...


class JobAgentQueryRequest(BaseModel):
    user_query: str = Field(min_length=1, max_length=1000)


class JobAgentQueryResponse(BaseModel):
    user_query: str
    planning_status: Literal["tool_selected", "refused"]
    planner_identity: str
    plan_reason: str
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    tool_error: dict[str, Any] | None


def build_job_agent_router(
    runtime: JobAgentRuntimeProvider | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1/job-agent", tags=["job-agent"])

    @router.post(
        "/query",
        response_model=JobAgentQueryResponse,
        summary="Plan and execute a current-job Agent request",
    )
    def query_job_agent(
        payload: JobAgentQueryRequest,
        request: Request,
    ) -> JobAgentQueryResponse:
        active_runtime = runtime or getattr(
            request.app.state,
            "job_agent_runtime",
            None,
        )
        if active_runtime is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job Agent is disabled or not prepared",
            )
        try:
            state = active_runtime.invoke(payload.user_query)
        except JobAgentPlannerRateLimitError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Job Agent planner is temporarily rate limited",
            ) from error
        except (
            JobAgentPlannerAuthenticationError,
            JobAgentPlannerUnavailableError,
        ) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job Agent planner is temporarily unavailable",
            ) from error
        except (
            JobAgentPlanningOutputDecodingError,
            JobAgentPlannerResponseError,
        ) as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Job Agent planner returned an invalid response",
            ) from error
        except JobAgentPlannerUpstreamError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Job Agent planner upstream request failed",
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        return JobAgentQueryResponse(
            user_query=payload.user_query,
            planning_status=state["planning_status"],
            planner_identity=state["planner_identity"],
            plan_reason=state["plan_reason"],
            tool_call=state.get("tool_call"),
            tool_result=state.get("tool_result"),
            tool_error=state.get("tool_error"),
        )

    return router
