from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Callable

from app.application.jobs.job_agent_planner_output_decoding import (
    JobAgentPlanningOutputDecoder,
)
from app.application.jobs.job_agent_planner_prompting import (
    JOB_AGENT_PLANNER_PROMPT_VERSION,
    JobAgentPlanningPromptBuilder,
)
from app.application.jobs.job_agent_planning import (
    AgentToolDefinition,
    JobAgentPlan,
    JobAgentPlannerTokenUsage,
    ObservedJobAgentPlan,
)


class JobAgentPlannerConfigurationError(ValueError):
    pass


class JobAgentPlannerResponseError(RuntimeError):
    pass


class JobAgentPlannerUpstreamError(RuntimeError):
    """Stable boundary error for failures reported by the model provider."""


class JobAgentPlannerAuthenticationError(JobAgentPlannerUpstreamError):
    pass


class JobAgentPlannerRateLimitError(JobAgentPlannerUpstreamError):
    pass


class JobAgentPlannerUnavailableError(JobAgentPlannerUpstreamError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleJobAgentPlannerConfig:
    model_name: str
    base_url: str
    api_key_environment_variable: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 30.0
    max_completion_tokens: int = 800
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise JobAgentPlannerConfigurationError(
                "Job Agent planner model_name must not be blank"
            )
        if not self.base_url.strip().startswith("https://"):
            raise JobAgentPlannerConfigurationError(
                "Job Agent planner base_url must use https"
            )
        if not self.api_key_environment_variable.strip():
            raise JobAgentPlannerConfigurationError(
                "API key environment variable must not be blank"
            )
        if self.timeout_seconds <= 0:
            raise JobAgentPlannerConfigurationError(
                "Job Agent planner timeout must be positive"
            )
        if self.max_completion_tokens < 1:
            raise JobAgentPlannerConfigurationError(
                "max_completion_tokens must be positive"
            )


class OpenAiCompatibleJobAgentPlanner:
    def __init__(
        self,
        completion_create: Callable[..., object],
        config: OpenAiCompatibleJobAgentPlannerConfig,
        *,
        call_id_factory: Callable[[], str] | None = None,
        prompt_builder: JobAgentPlanningPromptBuilder | None = None,
        output_decoder: JobAgentPlanningOutputDecoder | None = None,
    ) -> None:
        self._completion_create = completion_create
        self._config = config
        self._call_id_factory = call_id_factory or (
            lambda: f"call_{uuid.uuid4().hex}"
        )
        self._prompt_builder = (
            prompt_builder or JobAgentPlanningPromptBuilder()
        )
        self._output_decoder = (
            output_decoder or JobAgentPlanningOutputDecoder()
        )
        self._identity = (
            f"openai-compatible;model={config.model_name.strip()};"
            "temperature=0;response-format=json-object;"
            f"enable-thinking={str(config.enable_thinking).lower()};"
            f"prompt={JOB_AGENT_PLANNER_PROMPT_VERSION}"
        )

    @property
    def identity(self) -> str:
        return self._identity

    def plan(
        self,
        *,
        user_query: str,
        tools: tuple[AgentToolDefinition, ...],
    ) -> JobAgentPlan:
        return self.plan_observed(
            user_query=user_query,
            tools=tools,
        ).plan

    def plan_observed(
        self,
        *,
        user_query: str,
        tools: tuple[AgentToolDefinition, ...],
    ) -> ObservedJobAgentPlan:
        prompt = self._prompt_builder.build(
            user_query=user_query,
            tools=tools,
        )
        try:
            response = self._completion_create(
                model=self._config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompt.system_instruction,
                    },
                    {
                        "role": "user",
                        "content": prompt.user_payload,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=self._config.max_completion_tokens,
                extra_body={
                    "enable_thinking": self._config.enable_thinking
                },
            )
        except Exception as error:
            translated_error = _translate_provider_error(error)
            if translated_error is None:
                raise
            raise translated_error from error
        try:
            content = response.choices[0].message.content  # type: ignore[attr-defined]
        except (AttributeError, IndexError, TypeError) as error:
            raise JobAgentPlannerResponseError(
                "Job Agent planner response has no message content"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise JobAgentPlannerResponseError(
                "Job Agent planner returned blank message content"
            )
        return ObservedJobAgentPlan(
            plan=self._output_decoder.decode(
                content,
                call_id=self._call_id_factory(),
                allowed_tool_names=tuple(tool.name for tool in tools),
            ),
            token_usage=self._read_token_usage(response),
        )

    @staticmethod
    def _read_token_usage(
        response: object,
    ) -> JobAgentPlannerTokenUsage | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        try:
            return JobAgentPlannerTokenUsage(
                input_tokens=int(usage.prompt_tokens),
                output_tokens=int(usage.completion_tokens),
                total_tokens=int(usage.total_tokens),
            )
        except (AttributeError, TypeError, ValueError):
            return None


def _translate_provider_error(error: Exception) -> RuntimeError | None:
    status_code = _provider_status_code(error)
    error_name = type(error).__name__.casefold()
    if status_code in (401, 403) or error_name in {
        "authenticationerror",
        "permissiondeniederror",
    }:
        return JobAgentPlannerAuthenticationError(
            "Job Agent planner authentication failed"
        )
    if status_code == 429 or error_name == "ratelimiterror":
        return JobAgentPlannerRateLimitError(
            "Job Agent planner rate limit was reached"
        )
    if (
        status_code is not None
        and status_code >= 500
        or error_name
        in {
            "apiconnectionerror",
            "apitimeouterror",
            "internalservererror",
        }
    ):
        return JobAgentPlannerUnavailableError(
            "Job Agent planner provider is unavailable"
        )
    if status_code is not None:
        return JobAgentPlannerUpstreamError(
            "Job Agent planner provider rejected the request"
        )
    return None


def _provider_status_code(error: Exception) -> int | None:
    direct_status = getattr(error, "status_code", None)
    if isinstance(direct_status, int):
        return direct_status
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def build_openai_compatible_job_agent_planner(
    config: OpenAiCompatibleJobAgentPlannerConfig,
) -> OpenAiCompatibleJobAgentPlanner:
    api_key = os.getenv(config.api_key_environment_variable)
    if not api_key:
        raise JobAgentPlannerConfigurationError(
            "Job Agent planner API key environment variable is not configured"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise JobAgentPlannerConfigurationError(
            "install the generation dependency before using the planner"
        ) from error

    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    return OpenAiCompatibleJobAgentPlanner(
        client.chat.completions.create,
        config,
    )
