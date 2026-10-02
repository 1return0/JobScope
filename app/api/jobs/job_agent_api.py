from __future__ import annotations

from typing import Any, Literal, Protocol

from datetime import datetime, timezone

from fastapi import APIRouter, Cookie, HTTPException, Request, Response, status
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
from app.application.jobs.job_agent_sessions import (
    IssuedJobAgentSession,
    JobAgentSessionNotFoundError,
)
from app.application.memory.long_term_memory import LongTermMemoryEntry
from app.application.jobs.agent_harness import (
    JobAgentHarnessSnapshot,
    JobAgentHarnessLimitExceededError,
)


JOB_AGENT_SESSION_COOKIE_NAME = "jobscope_job_agent_session"


class JobAgentRuntimeProvider(Protocol):
    def invoke(self, user_query: str, *, owner_id: str) -> dict[str, Any]:
        ...

    def issue_browser_session(self) -> IssuedJobAgentSession:
        ...

    def resolve_browser_owner(self, session_token: str | None) -> str:
        ...

    def remember_preference(
        self,
        *,
        owner_id: str,
        preference_key: str,
        preference_value: str,
        user_consented: bool,
    ) -> LongTermMemoryEntry:
        ...

    def list_preferences(
        self,
        *,
        owner_id: str,
    ) -> tuple[LongTermMemoryEntry, ...]:
        ...

    def forget_preference(
        self,
        *,
        owner_id: str,
        preference_key: str,
    ) -> bool:
        ...

    def forget_all_preferences(self, *, owner_id: str) -> int:
        ...

    def get_conversation_turns(
        self,
        *,
        owner_id: str,
    ) -> tuple[dict[str, str], ...]:
        ...

    def clear_conversation_turns(self, *, owner_id: str) -> bool:
        ...

    @property
    def session_cookie_secure(self) -> bool:
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
    harness: "JobAgentHarnessResponse | None" = None


class JobAgentHarnessStepResponse(BaseModel):
    operation_kind: Literal["model", "tool"]
    operation_name: str
    status: Literal["succeeded", "failed", "rejected"]
    elapsed_seconds: float
    failure_code: str | None


class JobAgentHarnessResponse(BaseModel):
    model_call_count: int
    tool_call_count: int
    elapsed_seconds: float
    steps: list[JobAgentHarnessStepResponse]


class JobAgentSessionResponse(BaseModel):
    expires_at: datetime


class JobAgentPreferenceWriteRequest(BaseModel):
    preference_key: Literal[
        "preferred_location",
        "preferred_recruitment_type",
    ]
    preference_value: str = Field(min_length=1, max_length=100)
    user_consented: Literal[True]


class JobAgentPreferenceResponse(BaseModel):
    preference_key: str
    preference_value: str
    consented_at: datetime
    created_at: datetime
    updated_at: datetime
    revision: int


class JobAgentPreferencesResponse(BaseModel):
    preferences: list[JobAgentPreferenceResponse]


class JobAgentForgetAllPreferencesResponse(BaseModel):
    deleted_count: int


class JobAgentConversationTurnResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class JobAgentConversationResponse(BaseModel):
    turns: list[JobAgentConversationTurnResponse]


def build_job_agent_router(
    runtime: JobAgentRuntimeProvider | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1/job-agent", tags=["job-agent"])

    @router.post(
        "/sessions",
        response_model=JobAgentSessionResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Issue an opaque browser session for the Job Agent",
    )
    def create_job_agent_session(
        request: Request,
        response: Response,
    ) -> JobAgentSessionResponse:
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
        issued = active_runtime.issue_browser_session()
        max_age = max(
            1,
            int((issued.expires_at - datetime.now(timezone.utc)).total_seconds()),
        )
        response.set_cookie(
            key=JOB_AGENT_SESSION_COOKIE_NAME,
            value=issued.session_token,
            max_age=max_age,
            httponly=True,
            secure=active_runtime.session_cookie_secure,
            samesite="lax",
            path="/v1/job-agent",
        )
        return JobAgentSessionResponse(expires_at=issued.expires_at)

    @router.post(
        "/query",
        response_model=JobAgentQueryResponse,
        summary="Plan and execute a current-job Agent request",
    )
    def query_job_agent(
        payload: JobAgentQueryRequest,
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=JOB_AGENT_SESSION_COOKIE_NAME,
        ),
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
            owner_id = active_runtime.resolve_browser_owner(session_token)
            state = active_runtime.invoke(
                payload.user_query,
                owner_id=owner_id,
            )
        except JobAgentSessionNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="A valid Job Agent session is required",
            ) from error
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
        except JobAgentHarnessLimitExceededError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error.failure_code,
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
            harness=_to_harness_response(state.get("harness_snapshot")),
        )

    @router.get(
        "/memory/preferences",
        response_model=JobAgentPreferencesResponse,
        summary="List the current browser owner's consented Job Agent preferences",
    )
    def list_job_agent_preferences(
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=JOB_AGENT_SESSION_COOKIE_NAME,
        ),
    ) -> JobAgentPreferencesResponse:
        active_runtime = _require_runtime(runtime, request)
        owner_id = _resolve_owner(active_runtime, session_token)
        return JobAgentPreferencesResponse(
            preferences=[
                _to_preference_response(entry)
                for entry in active_runtime.list_preferences(owner_id=owner_id)
            ]
        )

    @router.put(
        "/memory/preferences",
        response_model=JobAgentPreferenceResponse,
        summary="Save one explicitly consented Job Agent preference",
    )
    def remember_job_agent_preference(
        payload: JobAgentPreferenceWriteRequest,
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=JOB_AGENT_SESSION_COOKIE_NAME,
        ),
    ) -> JobAgentPreferenceResponse:
        active_runtime = _require_runtime(runtime, request)
        owner_id = _resolve_owner(active_runtime, session_token)
        try:
            entry = active_runtime.remember_preference(
                owner_id=owner_id,
                preference_key=payload.preference_key,
                preference_value=payload.preference_value,
                user_consented=payload.user_consented,
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job Agent long-term memory is not prepared",
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        return _to_preference_response(entry)

    @router.delete(
        "/memory/preferences/{preference_key}",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Delete one consented Job Agent preference",
    )
    def forget_job_agent_preference(
        preference_key: Literal[
            "preferred_location",
            "preferred_recruitment_type",
        ],
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=JOB_AGENT_SESSION_COOKIE_NAME,
        ),
    ) -> Response:
        active_runtime = _require_runtime(runtime, request)
        owner_id = _resolve_owner(active_runtime, session_token)
        try:
            active_runtime.forget_preference(
                owner_id=owner_id,
                preference_key=preference_key,
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job Agent long-term memory is not prepared",
            ) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.delete(
        "/memory/preferences",
        response_model=JobAgentForgetAllPreferencesResponse,
        summary="Delete all consented Job Agent preferences for the current owner",
    )
    def forget_all_job_agent_preferences(
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=JOB_AGENT_SESSION_COOKIE_NAME,
        ),
    ) -> JobAgentForgetAllPreferencesResponse:
        active_runtime = _require_runtime(runtime, request)
        owner_id = _resolve_owner(active_runtime, session_token)
        try:
            deleted_count = active_runtime.forget_all_preferences(
                owner_id=owner_id
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job Agent long-term memory is not prepared",
            ) from error
        return JobAgentForgetAllPreferencesResponse(
            deleted_count=deleted_count
        )

    @router.get(
        "/conversation",
        response_model=JobAgentConversationResponse,
        summary="Read the current browser owner's bounded Job Agent conversation",
    )
    def get_job_agent_conversation(
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=JOB_AGENT_SESSION_COOKIE_NAME,
        ),
    ) -> JobAgentConversationResponse:
        active_runtime = _require_runtime(runtime, request)
        owner_id = _resolve_owner(active_runtime, session_token)
        try:
            turns = active_runtime.get_conversation_turns(owner_id=owner_id)
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job Agent conversation memory is not prepared",
            ) from error
        return JobAgentConversationResponse(
            turns=[JobAgentConversationTurnResponse(**turn) for turn in turns]
        )

    @router.delete(
        "/conversation",
        status_code=status.HTTP_204_NO_CONTENT,
        summary="Delete the current browser owner's Job Agent checkpoint thread",
    )
    def clear_job_agent_conversation(
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=JOB_AGENT_SESSION_COOKIE_NAME,
        ),
    ) -> Response:
        active_runtime = _require_runtime(runtime, request)
        owner_id = _resolve_owner(active_runtime, session_token)
        try:
            active_runtime.clear_conversation_turns(owner_id=owner_id)
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job Agent conversation memory is not prepared",
            ) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router


def _to_harness_response(
    snapshot: object,
) -> JobAgentHarnessResponse | None:
    if not isinstance(snapshot, JobAgentHarnessSnapshot):
        return None
    return JobAgentHarnessResponse(
        model_call_count=snapshot.model_call_count,
        tool_call_count=snapshot.tool_call_count,
        elapsed_seconds=snapshot.elapsed_seconds,
        steps=[
            JobAgentHarnessStepResponse(
                operation_kind=step.operation_kind,
                operation_name=step.operation_name,
                status=step.status,
                elapsed_seconds=step.elapsed_seconds,
                failure_code=step.failure_code,
            )
            for step in snapshot.steps
        ],
    )


def _require_runtime(
    supplied_runtime: JobAgentRuntimeProvider | None,
    request: Request,
) -> JobAgentRuntimeProvider:
    active_runtime = supplied_runtime or getattr(
        request.app.state,
        "job_agent_runtime",
        None,
    )
    if active_runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job Agent is disabled or not prepared",
        )
    return active_runtime


def _resolve_owner(
    runtime: JobAgentRuntimeProvider,
    session_token: str | None,
) -> str:
    try:
        return runtime.resolve_browser_owner(session_token)
    except JobAgentSessionNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid Job Agent session is required",
        ) from error


def _to_preference_response(
    entry: LongTermMemoryEntry,
) -> JobAgentPreferenceResponse:
    return JobAgentPreferenceResponse(
        preference_key=entry.preference_key,
        preference_value=entry.preference_value,
        consented_at=entry.consented_at,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        revision=entry.revision,
    )
