from __future__ import annotations

import json

from app.application.answering.claim_support_evaluation import (
    ClaimSupportEvaluationInput,
)
from app.config import load_settings
from app.infrastructure.llm.openai_compatible_claim_support_judge import (
    OpenAiCompatibleClaimSupportJudgeConfig,
    build_openai_compatible_claim_support_judge,
)


def main() -> int:
    settings = load_settings(dotenv_override=True)
    judge = build_openai_compatible_claim_support_judge(
        OpenAiCompatibleClaimSupportJudgeConfig(
            model_name=settings.answer_model_name,
            base_url=settings.answer_model_base_url,
            api_key_environment_variable=(
                settings.answer_model_api_key_environment_variable
            ),
            timeout_seconds=settings.answer_model_timeout_seconds,
            max_completion_tokens=500,
        )
    )
    cases = (
        ClaimSupportEvaluationInput(
            claim_index=1,
            claim_text="该岗位工作地点是上海。",
            evidence_texts=("该岗位工作地点位于上海。",),
        ),
        ClaimSupportEvaluationInput(
            claim_index=2,
            claim_text="该岗位工作地点是上海。",
            evidence_texts=("该岗位工作地点位于北京。",),
        ),
        ClaimSupportEvaluationInput(
            claim_index=3,
            claim_text="该岗位工作地点是上海。",
            evidence_texts=("该岗位专业不限。",),
        ),
    )
    payload = []
    for case in cases:
        judgment = judge.judge(case)
        payload.append(
            {
                "claim_index": case.claim_index,
                "label": judgment.label,
                "rationale": judgment.rationale,
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
