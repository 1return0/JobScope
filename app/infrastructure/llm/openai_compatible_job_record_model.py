from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from app.application.jobs.job_record_prompting import (
    JobRecordExtractionPrompt,
)


class JobRecordModelConfigurationError(ValueError):
    pass


class JobRecordModelResponseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleJobRecordModelConfig:
    model_name: str
    base_url: str
    api_key_environment_variable: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 30.0
    max_completion_tokens: int = 2400
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise JobRecordModelConfigurationError(
                "job record model_name must not be blank"
            )
        if not self.base_url.strip().startswith("https://"):
            raise JobRecordModelConfigurationError(
                "job record model base_url must use https"
            )
        if not self.api_key_environment_variable.strip():
            raise JobRecordModelConfigurationError(
                "API key environment variable must not be blank"
            )
        if self.timeout_seconds <= 0:
            raise JobRecordModelConfigurationError(
                "job record model timeout must be positive"
            )
        if self.max_completion_tokens < 1:
            raise JobRecordModelConfigurationError(
                "max_completion_tokens must be positive"
            )


class OpenAiCompatibleJobRecordModel:
    def __init__(
        self,
        completion_create: Callable[..., object],
        config: OpenAiCompatibleJobRecordModelConfig,
    ) -> None:
        self._completion_create = completion_create
        self._config = config
        self._identity = (
            f"openai-compatible;model={config.model_name.strip()};"
            "temperature=0;response-format=json-object;"
            f"enable-thinking={str(config.enable_thinking).lower()}"
        )

    @property
    def identity(self) -> str:
        return self._identity

    def generate(self, prompt: JobRecordExtractionPrompt) -> str:
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
        try:
            content = response.choices[0].message.content  # type: ignore[attr-defined]
        except (AttributeError, IndexError, TypeError) as error:
            raise JobRecordModelResponseError(
                "job record model response has no message content"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise JobRecordModelResponseError(
                "job record model returned blank message content"
            )
        return content


def build_openai_compatible_job_record_model(
    config: OpenAiCompatibleJobRecordModelConfig,
) -> OpenAiCompatibleJobRecordModel:
    api_key = os.getenv(config.api_key_environment_variable)
    if not api_key:
        raise JobRecordModelConfigurationError(
            "job record model API key environment variable is not configured"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise JobRecordModelConfigurationError(
            "install the generation dependency before using the model"
        ) from error

    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    return OpenAiCompatibleJobRecordModel(
        client.chat.completions.create,
        config,
    )
