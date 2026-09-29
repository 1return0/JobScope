import unittest
from pathlib import Path

from app.application.jobs.job_agent_planner_evaluation_json import (
    JobAgentPlannerEvaluationDatasetError,
    decode_job_agent_planner_evaluation_dataset,
    load_job_agent_planner_evaluation_dataset,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "job-agent-planner-v1.json"
)


class JobAgentPlannerEvaluationJsonTest(unittest.TestCase):
    def test_loads_fixed_planner_dataset_and_computes_identity(self) -> None:
        dataset = load_job_agent_planner_evaluation_dataset(DATASET_PATH)

        self.assertEqual("job-agent-planner-v1", dataset.dataset_id)
        self.assertEqual("provisional", dataset.review_status)
        self.assertEqual(11, len(dataset.cases))
        agent_internship = next(
            case
            for case in dataset.cases
            if case.case_id == "search-agent-internship"
        )
        self.assertEqual("call_tool", agent_internship.expected_decision)
        self.assertEqual(
            {
                "job_title": "AI Agent",
                "recruitment_type": "internship",
            },
            agent_internship.expected_arguments,
        )
        self.assertEqual(64, len(dataset.dataset_sha256))

    def test_hash_is_stable_across_argument_key_order(self) -> None:
        first = self._payload()
        second = self._payload()
        second["cases"][0]["expected_arguments"] = {
            "limit": 5,
            "location": "北京",
        }

        first_dataset = decode_job_agent_planner_evaluation_dataset(first)
        second_dataset = decode_job_agent_planner_evaluation_dataset(second)

        self.assertEqual(
            first_dataset.dataset_sha256,
            second_dataset.dataset_sha256,
        )

    def test_hash_changes_when_expected_answer_changes(self) -> None:
        first = decode_job_agent_planner_evaluation_dataset(self._payload())
        changed_payload = self._payload()
        changed_payload["cases"][0]["expected_arguments"]["limit"] = 6
        changed = decode_job_agent_planner_evaluation_dataset(changed_payload)

        self.assertNotEqual(first.dataset_sha256, changed.dataset_sha256)

    def test_rejects_unknown_json_fields(self) -> None:
        payload = self._payload()
        payload["unexpected"] = True

        with self.assertRaises(JobAgentPlannerEvaluationDatasetError):
            decode_job_agent_planner_evaluation_dataset(payload)

    @staticmethod
    def _payload():
        return {
            "dataset_id": "planner-test-v1",
            "dataset_version": 1,
            "review_status": "provisional",
            "cases": [
                {
                    "case_id": "beijing-limit",
                    "user_query": "给我5个北京岗位",
                    "expected_decision": "call_tool",
                    "expected_tool_name": "search_current_jobs",
                    "expected_arguments": {
                        "location": "北京",
                        "limit": 5,
                    },
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
