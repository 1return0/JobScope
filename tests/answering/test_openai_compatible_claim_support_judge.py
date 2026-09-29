import unittest
from types import SimpleNamespace

from app.application.answering.claim_support_evaluation import (
    ClaimSupportEvaluationInput,
)
from app.infrastructure.llm.openai_compatible_claim_support_judge import (
    OpenAiCompatibleClaimSupportJudge,
    OpenAiCompatibleClaimSupportJudgeConfig,
)


class OpenAiCompatibleClaimSupportJudgeTest(unittest.TestCase):
    def test_sends_json_request_and_decodes_judgment(self) -> None:
        captured: dict[str, object] = {}

        def completion_create(**kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"label":"supported",'
                                '"rationale":"direct evidence"}'
                            )
                        )
                    )
                ]
            )

        judge = OpenAiCompatibleClaimSupportJudge(
            completion_create,
            OpenAiCompatibleClaimSupportJudgeConfig(
                model_name="test-model",
                base_url="https://example.com/v1",
            ),
        )

        result = judge.judge(
            ClaimSupportEvaluationInput(
                claim_index=1,
                claim_text="工作地点是上海。",
                evidence_texts=("工作地点位于上海。",),
            )
        )

        self.assertEqual("supported", result.label)
        self.assertEqual("test-model", captured["model"])
        self.assertEqual(0, captured["temperature"])
        self.assertEqual(
            {"type": "json_object"},
            captured["response_format"],
        )


if __name__ == "__main__":
    unittest.main()
