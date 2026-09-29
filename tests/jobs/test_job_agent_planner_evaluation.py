import unittest

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerEvaluationCase,
    JobAgentPlannerEvaluator,
    PydanticAgentToolArgumentsNormalizer,
    validate_job_agent_planner_evaluation_cases,
)
from app.application.jobs.job_agent_planning import (
    AgentToolDefinition,
    JobAgentPlan,
    JobAgentPlannerTokenUsage,
    ObservedJobAgentPlan,
)
from app.application.jobs.job_tool_graph import AgentToolCall
from app.application.jobs.job_search_tool import SearchCurrentJobsArguments


class _Planner:
    identity = "fake-planner-evaluation-v1"

    def __init__(self, plans: list[JobAgentPlan | Exception]) -> None:
        self._plans = iter(plans)
        self.calls = 0

    def plan(self, *, user_query, tools):
        self.calls += 1
        outcome = next(self._plans)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ObservablePlanner:
    identity = "observable-planner-v1"

    def plan_observed(self, *, user_query, tools):
        return ObservedJobAgentPlan(
            plan=JobAgentPlan(
                decision="refuse",
                reason="Out of scope.",
            ),
            token_usage=JobAgentPlannerTokenUsage(
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
            ),
        )


def _tools() -> tuple[AgentToolDefinition, ...]:
    return (
        AgentToolDefinition(
            name="search_current_jobs",
            description="Search current jobs.",
            arguments_schema={"type": "object"},
        ),
    )


class JobAgentPlannerEvaluatorTest(unittest.TestCase):
    def test_reports_decision_tool_argument_and_case_accuracy(self) -> None:
        planner = _Planner(
            [
                JobAgentPlan(
                    decision="call_tool",
                    reason="Search is required.",
                    tool_call=AgentToolCall(
                        call_id="call-1",
                        name="search_current_jobs",
                        arguments={"location": "上海"},
                    ),
                ),
                JobAgentPlan(
                    decision="refuse",
                    reason="Out of scope.",
                ),
                JobAgentPlan(
                    decision="call_tool",
                    reason="Search is required.",
                    tool_call=AgentToolCall(
                        call_id="call-3",
                        name="search_current_jobs",
                        arguments={"location": "北京"},
                    ),
                ),
            ]
        )
        evaluator = JobAgentPlannerEvaluator(planner, tools=_tools())

        report = evaluator.evaluate(
            (
                JobAgentPlannerEvaluationCase(
                    case_id="shanghai-search",
                    user_query="找上海岗位",
                    expected_decision="call_tool",
                    expected_tool_name="search_current_jobs",
                    expected_arguments={"location": "上海"},
                ),
                JobAgentPlannerEvaluationCase(
                    case_id="lottery-refusal",
                    user_query="预测彩票",
                    expected_decision="refuse",
                ),
                JobAgentPlannerEvaluationCase(
                    case_id="beijing-limit",
                    user_query="找5个北京岗位",
                    expected_decision="call_tool",
                    expected_tool_name="search_current_jobs",
                    expected_arguments={"location": "北京", "limit": 5},
                ),
            )
        )

        self.assertEqual(3, report.total_model_calls)
        self.assertEqual(3, report.successful_model_calls)
        self.assertEqual(0, report.failed_model_calls)
        self.assertEqual(1.0, report.decision_accuracy)
        self.assertEqual(1.0, report.tool_name_accuracy)
        self.assertEqual(0.5, report.arguments_accuracy)
        self.assertAlmostEqual(2 / 3, report.case_accuracy)
        self.assertFalse(report.results[2].passed)

    def test_rejects_duplicate_case_ids(self) -> None:
        planner = _Planner([])
        evaluator = JobAgentPlannerEvaluator(planner, tools=_tools())
        case = JobAgentPlannerEvaluationCase(
            case_id="duplicate",
            user_query="预测彩票",
            expected_decision="refuse",
        )

        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            evaluator.evaluate((case, case))

        self.assertEqual(0, planner.calls)

    def test_normalizes_omitted_and_explicit_tool_defaults(self) -> None:
        planner = _Planner(
            [
                JobAgentPlan(
                    decision="call_tool",
                    reason="Search is required.",
                    tool_call=AgentToolCall(
                        call_id="call-default",
                        name="search_current_jobs",
                        arguments={"location": "上海", "limit": 10},
                    ),
                )
            ]
        )
        evaluator = JobAgentPlannerEvaluator(
            planner,
            tools=_tools(),
            argument_normalizers={
                "search_current_jobs": PydanticAgentToolArgumentsNormalizer(
                    SearchCurrentJobsArguments
                )
            },
        )

        report = evaluator.evaluate(
            (
                JobAgentPlannerEvaluationCase(
                    case_id="default-limit",
                    user_query="找上海岗位",
                    expected_decision="call_tool",
                    expected_tool_name="search_current_jobs",
                    expected_arguments={"location": "上海"},
                ),
            )
        )

        self.assertTrue(report.results[0].arguments_correct)
        self.assertTrue(report.results[0].passed)
        self.assertIsNone(report.results[0].arguments_error_code)

    def test_marks_invalid_actual_arguments_without_aborting_batch(self) -> None:
        planner = _Planner(
            [
                JobAgentPlan(
                    decision="call_tool",
                    reason="Search is required.",
                    tool_call=AgentToolCall(
                        call_id="call-invalid",
                        name="search_current_jobs",
                        arguments={"location": "上海", "limit": 1000},
                    ),
                )
            ]
        )
        evaluator = JobAgentPlannerEvaluator(
            planner,
            tools=_tools(),
            argument_normalizers={
                "search_current_jobs": PydanticAgentToolArgumentsNormalizer(
                    SearchCurrentJobsArguments
                )
            },
        )

        report = evaluator.evaluate(
            (
                JobAgentPlannerEvaluationCase(
                    case_id="invalid-limit",
                    user_query="找上海岗位",
                    expected_decision="call_tool",
                    expected_tool_name="search_current_jobs",
                    expected_arguments={"location": "上海"},
                ),
            )
        )

        self.assertFalse(report.results[0].arguments_correct)
        self.assertFalse(report.results[0].passed)
        self.assertEqual(
            "actual-arguments-invalid",
            report.results[0].arguments_error_code,
        )

    def test_rejects_invalid_expected_arguments_before_model_calls(self) -> None:
        planner = _Planner([])
        evaluator = JobAgentPlannerEvaluator(
            planner,
            tools=_tools(),
            argument_normalizers={
                "search_current_jobs": PydanticAgentToolArgumentsNormalizer(
                    SearchCurrentJobsArguments
                )
            },
        )

        with self.assertRaisesRegex(
            ValueError,
            "invalid expected arguments.*invalid-golden-case",
        ):
            evaluator.evaluate(
                (
                    JobAgentPlannerEvaluationCase(
                        case_id="invalid-golden-case",
                        user_query="找1000个上海岗位",
                        expected_decision="call_tool",
                        expected_tool_name="search_current_jobs",
                        expected_arguments={
                            "location": "上海",
                            "limit": 1000,
                        },
                    ),
                )
            )

        self.assertEqual(0, planner.calls)

    def test_preflight_reports_case_mix_without_model_calls(self) -> None:
        report = validate_job_agent_planner_evaluation_cases(
            (
                JobAgentPlannerEvaluationCase(
                    case_id="search",
                    user_query="Find AI Agent internships",
                    expected_decision="call_tool",
                    expected_tool_name="search_current_jobs",
                    expected_arguments={
                        "job_title": "AI Agent",
                        "recruitment_type": "internship",
                    },
                ),
                JobAgentPlannerEvaluationCase(
                    case_id="refuse",
                    user_query="Predict lottery numbers",
                    expected_decision="refuse",
                ),
            ),
            tools=_tools(),
            argument_normalizers={
                "search_current_jobs": PydanticAgentToolArgumentsNormalizer(
                    SearchCurrentJobsArguments
                )
            },
        )

        self.assertEqual(2, report.total_cases)
        self.assertEqual(1, report.call_tool_cases)
        self.assertEqual(1, report.refusal_cases)
        self.assertEqual(("search_current_jobs",), report.registered_tool_names)

    def test_preflight_rejects_unregistered_expected_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "unregistered tool"):
            validate_job_agent_planner_evaluation_cases(
                (
                    JobAgentPlannerEvaluationCase(
                        case_id="bad-tool",
                        user_query="Delete everything",
                        expected_decision="call_tool",
                        expected_tool_name="delete_all_jobs",
                        expected_arguments={},
                    ),
                ),
                tools=_tools(),
            )

    def test_isolates_planner_failure_and_continues_remaining_cases(self) -> None:
        planner = _Planner(
            [
                ConnectionError("temporary model connection failure"),
                JobAgentPlan(
                    decision="refuse",
                    reason="Out of scope.",
                ),
            ]
        )
        evaluator = JobAgentPlannerEvaluator(planner, tools=_tools())

        report = evaluator.evaluate(
            (
                JobAgentPlannerEvaluationCase(
                    case_id="connection-failure",
                    user_query="Find Shanghai jobs",
                    expected_decision="call_tool",
                    expected_tool_name="search_current_jobs",
                    expected_arguments={"location": "Shanghai"},
                ),
                JobAgentPlannerEvaluationCase(
                    case_id="refusal-after-failure",
                    user_query="Predict lottery numbers",
                    expected_decision="refuse",
                ),
            )
        )

        self.assertEqual(2, planner.calls)
        self.assertEqual(1, report.successful_model_calls)
        self.assertEqual(1, report.failed_model_calls)
        self.assertEqual("planner_failed", report.results[0].execution_status)
        self.assertEqual(
            "planner-call-failed",
            report.results[0].planner_error_code,
        )
        self.assertEqual("ConnectionError", report.results[0].planner_error_type)
        self.assertFalse(report.results[0].passed)
        self.assertTrue(report.results[1].passed)

    def test_records_case_latency_and_reports_progress(self) -> None:
        planner = _Planner(
            [
                JobAgentPlan(
                    decision="refuse",
                    reason="Out of scope.",
                ),
                JobAgentPlan(
                    decision="refuse",
                    reason="Out of scope.",
                ),
            ]
        )
        clock_values = iter((10.0, 10.25, 20.0, 20.75))
        progress_events = []
        evaluator = JobAgentPlannerEvaluator(
            planner,
            tools=_tools(),
            monotonic_clock=lambda: next(clock_values),
            progress_observer=progress_events.append,
        )

        report = evaluator.evaluate(
            (
                JobAgentPlannerEvaluationCase(
                    case_id="first",
                    user_query="Weather",
                    expected_decision="refuse",
                ),
                JobAgentPlannerEvaluationCase(
                    case_id="second",
                    user_query="Lottery",
                    expected_decision="refuse",
                ),
            )
        )

        self.assertEqual(0.25, report.results[0].elapsed_seconds)
        self.assertEqual(0.75, report.results[1].elapsed_seconds)
        self.assertEqual(1.0, report.total_case_elapsed_seconds)
        self.assertEqual(0.5, report.average_case_elapsed_seconds)
        self.assertEqual(1, progress_events[0].completed_cases)
        self.assertEqual(2, progress_events[0].total_cases)
        self.assertEqual("first", progress_events[0].case_id)
        self.assertEqual(2, progress_events[1].completed_cases)

    def test_aggregates_reported_token_usage(self) -> None:
        evaluator = JobAgentPlannerEvaluator(
            _ObservablePlanner(),
            tools=_tools(),
        )

        report = evaluator.evaluate(
            (
                JobAgentPlannerEvaluationCase(
                    case_id="usage",
                    user_query="Predict lottery numbers",
                    expected_decision="refuse",
                ),
            )
        )

        self.assertEqual(1, report.usage_reported_calls)
        self.assertEqual(0, report.usage_unavailable_calls)
        self.assertEqual(120, report.total_input_tokens)
        self.assertEqual(30, report.total_output_tokens)
        self.assertEqual(150, report.total_tokens)
        self.assertEqual(150, report.results[0].total_tokens)


if __name__ == "__main__":
    unittest.main()
