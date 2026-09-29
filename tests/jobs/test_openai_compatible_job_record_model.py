import unittest
from types import SimpleNamespace

from app.application.jobs.job_record_prompting import (
    JobRecordExtractionPrompt,
)
from app.infrastructure.llm.openai_compatible_job_record_model import (
    JobRecordModelConfigurationError,
    JobRecordModelResponseError,
    OpenAiCompatibleJobRecordModel,
    OpenAiCompatibleJobRecordModelConfig,
)


class OpenAiCompatibleJobRecordModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = OpenAiCompatibleJobRecordModelConfig(
            model_name="test-model",
            base_url="https://example.invalid/compatible-mode/v1",
        )
        self.prompt = JobRecordExtractionPrompt(
            system_instruction="system rules",
            user_payload='{"evidence":[]}',
            allowed_evidence_ids=("ev_" + "1" * 64,),
            prompt_version="job-record-extraction-v1",
            extraction_contract_version="job-record-v1",
        )

    def test_sends_role_separated_json_request_and_exposes_identity(
        self,
    ) -> None:
        captured = {}

        def completion_create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok":true}')
                    )
                ]
            )

        model = OpenAiCompatibleJobRecordModel(
            completion_create,
            self.config,
        )

        self.assertEqual('{"ok":true}', model.generate(self.prompt))
        self.assertEqual("system", captured["messages"][0]["role"])
        self.assertEqual("user", captured["messages"][1]["role"])
        self.assertEqual({"type": "json_object"}, captured["response_format"])
        self.assertEqual(0, captured["temperature"])
        self.assertEqual(
            {"enable_thinking": False},
            captured["extra_body"],
        )
        self.assertIn("model=test-model", model.identity)

    def test_rejects_missing_response_content(self) -> None:
        model = OpenAiCompatibleJobRecordModel(
            lambda **_: SimpleNamespace(choices=[]),
            self.config,
        )

        with self.assertRaises(JobRecordModelResponseError):
            model.generate(self.prompt)

    def test_rejects_insecure_base_url(self) -> None:
        with self.assertRaises(JobRecordModelConfigurationError):
            OpenAiCompatibleJobRecordModelConfig(
                model_name="test-model",
                base_url="http://example.invalid/v1",
            )


if __name__ == "__main__":
    unittest.main()
