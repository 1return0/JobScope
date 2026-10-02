import unittest

from app.application.jobs.job_agent_thread_identity import (
    build_job_agent_thread_id,
)


class JobAgentThreadIdentityTest(unittest.TestCase):
    def test_same_owner_always_maps_to_same_non_plaintext_thread_id(self) -> None:
        owner_id = "f0d4db53-214d-42fb-9f32-14a07c29adbb"

        thread_id = build_job_agent_thread_id(owner_id)

        self.assertEqual(thread_id, build_job_agent_thread_id(owner_id))
        self.assertTrue(thread_id.startswith("jobscope-owner-"))
        self.assertNotIn(owner_id, thread_id)

    def test_blank_owner_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "owner_id"):
            build_job_agent_thread_id("   ")
