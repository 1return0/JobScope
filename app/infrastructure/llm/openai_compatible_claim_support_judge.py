from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from app.application.answering.claim_support_evaluation import (
    ClaimSupportEvaluationInput,
    ClaimSupportJudgment,
)
from app.application.answering.claim_support_output_decoding import (
    ClaimSupportOutputDecoder,
)
from app.application.answering.claim_support_prompting import (
    ClaimSupportPromptBuilder,
)


class ClaimSupportJudgeConfigurationError(ValueError):
    pass


class ClaimSupportJudgeResponseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OpenAiCompatibleClaimSupportJudgeConfig:
    model_name: str
    base_url: str
    api_key_environment_variable: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 30.0
    max_completion_tokens: int = 500

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ClaimSupportJudgeConfigurationError(
                "claim-support model_name must not be blank"
            )
        if not self.base_url.strip().startswith("https://"):
            raise ClaimSupportJudgeConfigurationError(
                "claim-support base_url must use https"
            )
        if not self.api_key_environment_variable.strip():
            raise ClaimSupportJudgeConfigurationError(
                "claim-support API key variable must not be blank"
            )
        if self.timeout_seconds <= 0:
            raise ClaimSupportJudgeConfigurationError(
                "claim-support timeout must be positive"
            )
        if self.max_completion_tokens < 1:
            raise ClaimSupportJudgeConfigurationError(
                "claim-support max tokens must be positive"
            )


class OpenAiCompatibleClaimSupportJudge:
    def __init__(
        self,
        completion_create: Callable[..., object],
        config: OpenAiCompatibleClaimSupportJudgeConfig,
        *,
        prompt_builder: ClaimSupportPromptBuilder | None = None,
        output_decoder: ClaimSupportOutputDecoder | None = None,
    ) -> None:
        self._completion_create = completion_create
        self._config = config
        self._prompt_builder = prompt_builder or ClaimSupportPromptBuilder()
        self._output_decoder = output_decoder or ClaimSupportOutputDecoder()

    def judge(
        self,
        evaluation_input: ClaimSupportEvaluationInput,
    ) -> ClaimSupportJudgment:
        prompt = self._prompt_builder.build(evaluation_input)
        response = self._completion_create(
            model=self._config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": prompt.system_message,
                },
                {
                    "role": "user",
                    "content": prompt.user_message,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=self._config.max_completion_tokens,
        )
        try:
            content = response.choices[0].message.content  # type: ignore[attr-defined]
        except (AttributeError, IndexError, TypeError) as error:
            raise ClaimSupportJudgeResponseError(
                "claim-support model response has no message content"
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise ClaimSupportJudgeResponseError(
                "claim-support model returned blank content"
            )
        return self._output_decoder.decode(content)


def build_openai_compatible_claim_support_judge(
    config: OpenAiCompatibleClaimSupportJudgeConfig,
) -> OpenAiCompatibleClaimSupportJudge:
    api_key = os.getenv(config.api_key_environment_variable)
    if not api_key:
        raise ClaimSupportJudgeConfigurationError(
            "claim-support API key environment variable is not configured"
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ClaimSupportJudgeConfigurationError(
            "install the generation dependency group before using the judge"
        ) from error

    client = OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    return OpenAiCompatibleClaimSupportJudge(
        client.chat.completions.create,
        config,
    )
