from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from app.application.jobs.job_fact_support_output_decoding import (
    JobFactSupportOutputDecoder,
)
from app.application.jobs.job_fact_support_prompting import (
    JobFactSupportPromptBuilder,
)
from app.application.jobs.job_record_semantic_support import (
    JobFactSupportDecision,
)
from app.domain.jobs.structured_job_record import EvidenceBackedJobFact


class JobFactSupportJudgeConfigurationError(ValueError):
    pass


class JobFactSupportJudgeResponseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleJobFactSupportJudgeConfig:
    model_name: str
    base_url: str
    api_key_environment_variable: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 30.0
    max_completion_tokens: int = 500
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise JobFactSupportJudgeConfigurationError(
                "job fact support model_name must not be blank"
            )
        if not self.base_url.strip().startswith("https://"):
            raise JobFactSupportJudgeConfigurationError(
                "job fact support base_url must use https"
            )
        if not self.api_key_environment_variable.strip():
            raise JobFactSupportJudgeConfigurationError(
                "API key environment variable must not be blank"
            )
        if self.timeout_seconds <= 0:
            raise JobFactSupportJudgeConfigurationError(
                "job fact support timeout must be positive"
            )
        if self.max_completion_tokens < 1:
            raise JobFactSupportJudgeConfigurationError(
                "max_completion_tokens must be positive"
            )


class OpenAiCompatibleJobFactSupportJudge:
    def __init__(
        self,
        completion_create: Callable[..., object],
        config: OpenAiCompatibleJobFactSupportJudgeConfig,
        *,
        prompt_builder: JobFactSupportPromptBuilder | None = None,
        output_decoder: JobFactSupportOutputDecoder | None = None,
    ) -> None:
        self._completion_create = completion_create
        self._config = config
        self._prompt_builder = prompt_builder or JobFactSupportPromptBuilder()
        self._output_decoder = output_decoder or JobFactSupportOutputDecoder()
        self._identity = (
            f"openai-compatible;role=job-fact-support;"
            f"model={config.model_name.strip()};temperature=0;"
            f"enable-thinking={str(config.enable_thinking).lower()}"
        )

    @property
    def identity(self) -> str:
        return self._identity

    def judge(
        self,
        *,
        field_path: str,
        fact: EvidenceBackedJobFact,
    ) -> JobFactSupportDecision:
        prompt = self._prompt_builder.build(
            field_path=field_path,
            fact=fact,
        )
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
            raise JobFactSupportJudgeResponseError(
                "job fact support response has no message content"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise JobFactSupportJudgeResponseError(
                "job fact support model returned blank message content"
            )
        return self._output_decoder.decode(content)


def build_openai_compatible_job_fact_support_judge(
    config: OpenAiCompatibleJobFactSupportJudgeConfig,
) -> OpenAiCompatibleJobFactSupportJudge:
    api_key = os.getenv(config.api_key_environment_variable)
    if not api_key:
        raise JobFactSupportJudgeConfigurationError(
            "job fact support API key environment variable is not configured"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise JobFactSupportJudgeConfigurationError(
            "install the generation dependency before using the judge"
        ) from error
    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    return OpenAiCompatibleJobFactSupportJudge(
        client.chat.completions.create,
        config,
    )
