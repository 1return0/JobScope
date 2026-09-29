import unittest

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerEvaluationReport,
)
from app.application.jobs.job_agent_planner_evaluation_gate import (
    JobAgentPlannerEvaluationGatePolicy,
    evaluate_job_agent_planner_report_gate,
)


def _report(**overrides) -> JobAgentPlannerEvaluationReport:
    values = {
        "total_cases": 10,
        "total_model_calls": 10,
        "successful_model_calls": 10,
        "failed_model_calls": 0,
        "total_case_elapsed_seconds": 20.0,
        "average_case_elapsed_seconds": 2.0,
        "usage_reported_calls": 10,
        "usage_unavailable_calls": 0,
        "total_input_tokens": 5000,
        "total_output_tokens": 1000,
        "total_tokens": 6000,
        "decision_accuracy": 1.0,
        "tool_name_accuracy": 1.0,
        "arguments_accuracy": 1.0,
        "case_accuracy": 1.0,
        "planner_identity": "planner-test",
        "results": (),
    }
    values.update(overrides)
    return JobAgentPlannerEvaluationReport(**values)


class JobAgentPlannerEvaluationGateTest(unittest.TestCase):
    def test_passes_report_within_all_thresholds(self) -> None:
        result = evaluate_job_agent_planner_report_gate(
            _report(),
            JobAgentPlannerEvaluationGatePolicy(),
        )

        self.assertTrue(result.passed)
        self.assertEqual(600.0, result.average_tokens_per_reported_call)
        self.assertEqual((), result.violations)

    def test_reports_each_violated_threshold(self) -> None:
        result = evaluate_job_agent_planner_report_gate(
            _report(
                case_accuracy=0.7,
                failed_model_calls=1,
                average_case_elapsed_seconds=6.0,
                usage_reported_calls=9,
                usage_unavailable_calls=1,
                total_tokens=10800,
            ),
            JobAgentPlannerEvaluationGatePolicy(),
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            {
                "case-accuracy-below-minimum",
                "model-failures-above-maximum",
                "average-latency-above-maximum",
                "usage-unavailable-above-maximum",
                "average-tokens-above-maximum",
            },
            {violation.code for violation in result.violations},
        )

    def test_rejects_invalid_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            JobAgentPlannerEvaluationGatePolicy(
                minimum_case_accuracy=1.1,
            )


if __name__ == "__main__":
    unittest.main()
