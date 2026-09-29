import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerCaseResult,
    JobAgentPlannerEvaluationCase,
    JobAgentPlannerEvaluationReport,
)
from app.application.jobs.job_agent_planner_evaluation_json import (
    JobAgentPlannerEvaluationDataset,
)
from app.application.jobs.job_agent_planner_evaluation_gate import (
    JobAgentPlannerEvaluationGatePolicy,
    evaluate_job_agent_planner_report_gate,
)
from app.infrastructure.reports.job_agent_planner_experiment_report_json import (
    save_job_agent_planner_experiment_report,
    serialize_job_agent_planner_experiment_report,
)


class JobAgentPlannerExperimentReportJsonTest(unittest.TestCase):
    def test_serializes_dataset_identity_metrics_and_case_details(self) -> None:
        dataset = JobAgentPlannerEvaluationDataset(
            dataset_id="planner-v1",
            dataset_version=1,
            review_status="provisional",
            cases=(
                JobAgentPlannerEvaluationCase(
                    case_id="shanghai-search",
                    user_query="find shanghai jobs",
                    expected_decision="call_tool",
                    expected_tool_name="search_current_jobs",
                    expected_arguments={"location": "Shanghai"},
                ),
            ),
        )
        report = JobAgentPlannerEvaluationReport(
            total_cases=1,
            total_model_calls=1,
            successful_model_calls=1,
            failed_model_calls=0,
            total_case_elapsed_seconds=0.25,
            average_case_elapsed_seconds=0.25,
            usage_reported_calls=1,
            usage_unavailable_calls=0,
            total_input_tokens=120,
            total_output_tokens=30,
            total_tokens=150,
            decision_accuracy=1.0,
            tool_name_accuracy=1.0,
            arguments_accuracy=1.0,
            case_accuracy=1.0,
            planner_identity="planner-test",
            results=(
                JobAgentPlannerCaseResult(
                    case_id="shanghai-search",
                    execution_status="completed",
                    decision_correct=True,
                    tool_name_correct=True,
                    arguments_correct=True,
                    passed=True,
                    actual_decision="call_tool",
                    actual_tool_name="search_current_jobs",
                    actual_arguments={"location": "Shanghai", "limit": 10},
                    arguments_error_code=None,
                    planner_error_code=None,
                    planner_error_type=None,
                    elapsed_seconds=0.25,
                    input_tokens=120,
                    output_tokens=30,
                    total_tokens=150,
                ),
            ),
        )

        payload = serialize_job_agent_planner_experiment_report(
            dataset,
            report,
            executed_at=datetime(
                2026,
                8,
                22,
                12,
                0,
                tzinfo=timezone.utc,
            ),
            gate_result=evaluate_job_agent_planner_report_gate(
                report,
                JobAgentPlannerEvaluationGatePolicy(),
            ),
        )

        self.assertEqual("planner-v1", payload["dataset_id"])
        self.assertEqual("provisional", payload["review_status"])
        self.assertEqual(dataset.dataset_sha256, payload["dataset_sha256"])
        self.assertEqual("planner-test", payload["planner_identity"])
        self.assertEqual(1, payload["successful_model_call_count"])
        self.assertEqual(0, payload["failed_model_call_count"])
        self.assertEqual(0.25, payload["timing"]["average_case_elapsed_seconds"])
        self.assertEqual(0.25, payload["cases"][0]["elapsed_seconds"])
        self.assertEqual(150, payload["token_usage"]["total_tokens"])
        self.assertTrue(payload["quality_gate"]["passed"])
        self.assertEqual([], payload["quality_gate"]["violations"])
        self.assertEqual(1.0, payload["metrics"]["case_accuracy"])
        self.assertEqual("shanghai-search", payload["cases"][0]["case_id"])
        self.assertIn("tool selection", payload["limitations"][1])

    def test_rejects_naive_execution_time(self) -> None:
        dataset = JobAgentPlannerEvaluationDataset(
            dataset_id="planner-v1",
            dataset_version=1,
            review_status="provisional",
            cases=(
                JobAgentPlannerEvaluationCase(
                    case_id="refuse",
                    user_query="tell me the weather",
                    expected_decision="refuse",
                ),
            ),
        )
        report = JobAgentPlannerEvaluationReport(
            total_cases=0,
            total_model_calls=0,
            successful_model_calls=0,
            failed_model_calls=0,
            total_case_elapsed_seconds=0.0,
            average_case_elapsed_seconds=0.0,
            usage_reported_calls=0,
            usage_unavailable_calls=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            decision_accuracy=0.0,
            tool_name_accuracy=None,
            arguments_accuracy=None,
            case_accuracy=0.0,
            planner_identity="planner-test",
            results=(),
        )

        with self.assertRaisesRegex(ValueError, "timezone"):
            serialize_job_agent_planner_experiment_report(
                dataset,
                report,
                executed_at=datetime(2026, 8, 22, 12, 0),
            )

    def test_saved_report_cannot_silently_overwrite_history(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "planner-report.json"
            save_job_agent_planner_experiment_report(output_path, {"run": 1})

            with self.assertRaises(FileExistsError):
                save_job_agent_planner_experiment_report(
                    output_path,
                    {"run": 2},
                )

            self.assertEqual(
                {"run": 1},
                json.loads(output_path.read_text(encoding="utf-8")),
            )


if __name__ == "__main__":
    unittest.main()
