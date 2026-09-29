import os
import unittest
from unittest.mock import patch

from app.config import load_settings


class JobAgentSettingsTest(unittest.TestCase):
    def test_loads_explicit_job_agent_enabled_flag(self) -> None:
        with patch.dict(
            os.environ,
            {"JOBSCOPE_JOB_AGENT_ENABLED": "true"},
            clear=False,
        ):
            settings = load_settings()

        self.assertTrue(settings.job_agent_enabled)


if __name__ == "__main__":
    unittest.main()
