import os
import unittest
from unittest.mock import patch

from app.config import load_settings


class AnswerModelSettingsTest(unittest.TestCase):
    def test_loads_generation_configuration_without_storing_secret(self) -> None:
        environment = {
            "JOBSCOPE_ANSWER_GENERATION_ENABLED": "true",
            "JOBSCOPE_ANSWER_MODEL_NAME": "test-model",
            "JOBSCOPE_ANSWER_MODEL_BASE_URL": "https://example.invalid/v1",
            "JOBSCOPE_ANSWER_MODEL_API_KEY_ENV": "TEST_MODEL_API_KEY",
            "JOBSCOPE_ANSWER_MODEL_TIMEOUT_SECONDS": "12.5",
            "JOBSCOPE_ANSWER_MODEL_MAX_COMPLETION_TOKENS": "700",
            "TEST_MODEL_API_KEY": "secret-not-loaded-into-settings",
        }
        with patch.dict(os.environ, environment, clear=False):
            settings = load_settings()

        self.assertTrue(settings.answer_generation_enabled)
        self.assertEqual("test-model", settings.answer_model_name)
        self.assertEqual(
            "TEST_MODEL_API_KEY",
            settings.answer_model_api_key_environment_variable,
        )
        self.assertEqual(12.5, settings.answer_model_timeout_seconds)
        self.assertEqual(700, settings.answer_model_max_completion_tokens)
        self.assertNotIn("secret-not-loaded", repr(settings))

    def test_dotenv_override_flag_is_explicitly_supported(self) -> None:
        settings = load_settings(dotenv_override=True)

        self.assertIsInstance(settings.answer_generation_enabled, bool)


if __name__ == "__main__":
    unittest.main()
