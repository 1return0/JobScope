import unittest

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerCaseResult,
    JobAgentPlannerEvaluationReport,
)
from app.application.jobs.job_agent_planner_experiment_comparison import (
    JobAgentPlannerExperimentRun,
    compare_job_agent_planner_experiments,
)


DATASET_SHA256 = "a" * 64


def _case(case_id: str, *, passed: bool) -> JobAgentPlannerCaseResult:
    return JobAgentPlannerCaseResult(
        case_id=case_id,
        execution_status="completed",
        decision_correct=passed,
        tool_name_correct=None,
        arguments_correct=None,
        passed=passed,
        actual_decision="refuse",
        actual_tool_name=None,
        actual_arguments=None,
        arguments_error_code=None,
        planner_error_code=None,
        planner_error_type=None,
        elapsed_seconds=2.0,
        input_tokens=500,
        output_tokens=100,
        total_tokens=600,
    )


def _report(
    *,
    identity: str = "planner-v1",
    accuracy: float = 1.0,
    latency: float = 2.0,
    total_tokens: int = 1200,
    results: tuple[JobAgentPlannerCaseResult, ...] | None = None,
) -> JobAgentPlannerEvaluationReport:
    selected_results = results or (
        _case("case-a", passed=True),
        _case("case-b", passed=True),
    )
    return JobAgentPlannerEvaluationReport(
        total_cases=2,
        total_model_calls=2,
        successful_model_calls=2,
        failed_model_calls=0,
        total_case_elapsed_seconds=latency * 2,
        average_case_elapsed_seconds=latency,
        usage_reported_calls=2,
        usage_unavailable_calls=0,
        total_input_tokens=1000,
        total_output_tokens=total_tokens - 1000,
        total_tokens=total_tokens,
        decision_accuracy=accuracy,
        tool_name_accuracy=None,
        arguments_accuracy=None,
        case_accuracy=accuracy,
        planner_identity=identity,
        results=selected_results,
    )


def _run(run_id: str, report: JobAgentPlannerEvaluationReport):
    return JobAgentPlannerExperimentRun(
        run_id=run_id,
        dataset_sha256=DATASET_SHA256,
        report=report,
    )


class JobAgentPlannerExperimentComparisonTest(unittest.TestCase):
    def test_compares_same_identity_regression_runs(self) -> None:
        comparison = compare_job_agent_planner_experiments(
            _run("baseline", _report()),
            _run(
                "candidate",
                _report(
                    accuracy=0.5,
                    latency=2.5,
                    total_tokens=1400,
                    results=(
                        _case("case-a", passed=True),
                        _case("case-b", passed=False),
                    ),
                ),
            ),
        )

        self.assertEqual(-0.5, comparison.case_accuracy_delta)
        self.assertEqual(0.5, comparison.average_latency_seconds_delta)
        self.assertEqual(100.0, comparison.average_tokens_delta)
        self.assertEqual(("case-b",), comparison.regressed_case_ids)
        self.assertEqual((), comparison.improved_case_ids)

    def test_rejects_dataset_or_identity_change_in_regression_mode(self) -> None:
        baseline = _run("baseline", _report())
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            compare_job_agent_planner_experiments(
                baseline,
                JobAgentPlannerExperimentRun(
                    run_id="candidate",
                    dataset_sha256="b" * 64,
                    report=_report(),
                ),
            )
        with self.assertRaisesRegex(ValueError, "planner identity"):
            compare_job_agent_planner_experiments(
                baseline,
                _run("candidate", _report(identity="planner-v2")),
            )

    def test_controlled_experiment_requires_declared_changed_factor(self) -> None:
        baseline = _run("baseline", _report())
        candidate = _run("candidate", _report(identity="planner-v2"))
        with self.assertRaisesRegex(ValueError, "changed_factor"):
            compare_job_agent_planner_experiments(
                baseline,
                candidate,
                comparison_kind="controlled_experiment",
            )

        comparison = compare_job_agent_planner_experiments(
            baseline,
            candidate,
            comparison_kind="controlled_experiment",
            changed_factor="model_name",
        )
        self.assertEqual("model_name", comparison.changed_factor)


if __name__ == "__main__":
    unittest.main()
