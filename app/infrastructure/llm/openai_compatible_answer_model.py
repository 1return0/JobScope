from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from app.application.answering.answer_prompting import GroundedAnswerPrompt


class AnswerModelConfigurationError(ValueError):
    pass


class AnswerModelResponseError(RuntimeError):
    pass


class AnswerModelUpstreamError(RuntimeError):
    """Stable boundary error for model-provider failures."""


class AnswerModelAuthenticationError(AnswerModelUpstreamError):
    pass


class AnswerModelRateLimitError(AnswerModelUpstreamError):
    pass


class AnswerModelUnavailableError(AnswerModelUpstreamError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleAnswerModelConfig:
    model_name: str
    base_url: str
    api_key_environment_variable: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 30.0
    max_completion_tokens: int = 1200

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise AnswerModelConfigurationError(
                "answer model_name must not be blank"
            )
        if not self.base_url.strip().startswith("https://"):
            raise AnswerModelConfigurationError(
                "answer model base_url must use https"
            )
        if not self.api_key_environment_variable.strip():
            raise AnswerModelConfigurationError(
                "api key environment variable must not be blank"
            )
        if self.timeout_seconds <= 0:
            raise AnswerModelConfigurationError(
                "answer model timeout must be positive"
            )
        if self.max_completion_tokens < 1:
            raise AnswerModelConfigurationError(
                "max_completion_tokens must be positive"
            )


class OpenAiCompatibleAnswerModel:
    def __init__(
        self,
        completion_create: Callable[..., object],
        config: OpenAiCompatibleAnswerModelConfig,
    ) -> None:
        self._completion_create = completion_create
        self._config = config

    def generate(self, prompt: GroundedAnswerPrompt) -> str:
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
                extra_body={"enable_thinking": False},
            )
        except Exception as error:
            translated_error = _translate_provider_error(error)
            if translated_error is None:
                raise
            raise translated_error from error
        try:
            content = response.choices[0].message.content  # type: ignore[attr-defined]
        except (AttributeError, IndexError, TypeError) as error:
            raise AnswerModelResponseError(
                "answer model response has no message content"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise AnswerModelResponseError(
                "answer model returned blank message content"
            )
        return content


def _translate_provider_error(error: Exception) -> RuntimeError | None:
    status_code = _provider_status_code(error)
    error_name = type(error).__name__.casefold()
    if status_code in (401, 403) or error_name in {
        "authenticationerror",
        "permissiondeniederror",
    }:
        return AnswerModelAuthenticationError(
            "answer model authentication failed"
        )
    if status_code == 429 or error_name == "ratelimiterror":
        return AnswerModelRateLimitError(
            "answer model rate limit was reached"
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
        return AnswerModelUnavailableError(
            "answer model provider is unavailable"
        )
    if status_code is not None:
        return AnswerModelUpstreamError(
            "answer model provider rejected the request"
        )
    return None


def _provider_status_code(error: Exception) -> int | None:
    direct_status = getattr(error, "status_code", None)
    if isinstance(direct_status, int):
        return direct_status
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def build_openai_compatible_answer_model(
    config: OpenAiCompatibleAnswerModelConfig,
) -> OpenAiCompatibleAnswerModel:
    api_key = os.getenv(config.api_key_environment_variable)
    if not api_key:
        raise AnswerModelConfigurationError(
            "answer model API key environment variable is not configured"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise AnswerModelConfigurationError(
            "install the generation dependency group before using the model"
        ) from error

    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    return OpenAiCompatibleAnswerModel(
        client.chat.completions.create,
        config,
    )
