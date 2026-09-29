import unittest
from types import SimpleNamespace

from app.application.answering.answer_prompting import GroundedAnswerPrompt
from app.infrastructure.llm.openai_compatible_answer_model import (
    AnswerModelConfigurationError,
    AnswerModelAuthenticationError,
    AnswerModelResponseError,
    OpenAiCompatibleAnswerModel,
    OpenAiCompatibleAnswerModelConfig,
)


class OpenAiCompatibleAnswerModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OpenAiCompatibleAnswerModelConfig(
            model_name="test-generation-model",
            base_url="https://example.invalid/compatible-mode/v1",
            max_completion_tokens=800,
        )
        self.prompt = GroundedAnswerPrompt(
            system_instruction="system rules",
            user_payload='{"question":"岗位要求是什么？"}',
            allowed_evidence_ids=("ev_" + "1" * 64,),
        )

    def test_sends_role_separated_messages_and_requests_json(self) -> None:
        captured: dict[str, object] = {}

        def completion_create(**kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"status":"insufficient_evidence",'
                            '"claims":[],"fallback_message":"证据不足"}'
                        )
                    )
                ]
            )

        model = OpenAiCompatibleAnswerModel(
            completion_create,
            self.config,
        )

        output = model.generate(self.prompt)

        self.assertIn('"status":"insufficient_evidence"', output)
        self.assertEqual("test-generation-model", captured["model"])
        self.assertEqual(
            {"type": "json_object"},
            captured["response_format"],
        )
        self.assertEqual(0, captured["temperature"])
        self.assertEqual(
            {"enable_thinking": False},
            captured["extra_body"],
        )
        messages = captured["messages"]
        self.assertEqual("system", messages[0]["role"])
        self.assertEqual("user", messages[1]["role"])

    def test_rejects_response_without_message_content(self) -> None:
        model = OpenAiCompatibleAnswerModel(
            lambda **_: SimpleNamespace(choices=[]),
            self.config,
        )

        with self.assertRaises(AnswerModelResponseError):
            model.generate(self.prompt)

    def test_config_rejects_insecure_base_url(self) -> None:
        with self.assertRaises(AnswerModelConfigurationError):
            OpenAiCompatibleAnswerModelConfig(
                model_name="test-model",
                base_url="http://example.invalid/v1",
            )

    def test_translates_authentication_failure(self) -> None:
        class AuthenticationError(Exception):
            status_code = 401

        def fail(**_: object) -> object:
            raise AuthenticationError("provider detail")

        model = OpenAiCompatibleAnswerModel(fail, self.config)

        with self.assertRaises(AnswerModelAuthenticationError):
            model.generate(self.prompt)


if __name__ == "__main__":
    unittest.main()
