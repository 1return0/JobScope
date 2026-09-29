from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, Protocol

from pydantic import BaseModel, ValidationError

from app.application.jobs.job_agent_planning import (
    AgentToolDefinition,
    JobAgentPlanner,
    JobAgentPlannerTokenUsage,
    ObservableJobAgentPlanner,
)


class AgentToolArgumentsNormalizationError(ValueError):
    pass


class AgentToolArgumentsNormalizer(Protocol):
    def normalize(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class PydanticAgentToolArgumentsNormalizer:
    model_type: type[BaseModel]

    def normalize(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            model = self.model_type.model_validate(arguments)
        except ValidationError as error:
            raise AgentToolArgumentsNormalizationError(
                "tool arguments do not match the registered model"
            ) from error
        return model.model_dump(mode="json", exclude_none=True)


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationCase:
    case_id: str
    user_query: str
    expected_decision: Literal["call_tool", "refuse"]
    expected_tool_name: str | None = None
    expected_arguments: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.user_query.strip():
            raise ValueError("evaluation case identity and query are required")
        if self.expected_decision == "call_tool":
            if not (self.expected_tool_name or "").strip():
                raise ValueError("call_tool case requires expected_tool_name")
            if self.expected_arguments is None:
                raise ValueError("call_tool case requires expected_arguments")
        elif (
            self.expected_tool_name is not None
            or self.expected_arguments is not None
        ):
            raise ValueError("refuse case must not expect tool data")


@dataclass(frozen=True, slots=True)
class JobAgentPlannerCaseResult:
    case_id: str
    execution_status: Literal["completed", "planner_failed"]
    decision_correct: bool
    tool_name_correct: bool | None
    arguments_correct: bool | None
    passed: bool
    actual_decision: str | None
    actual_tool_name: str | None
    actual_arguments: dict[str, Any] | None
    arguments_error_code: str | None
    planner_error_code: str | None
    planner_error_type: str | None
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationReport:
    total_cases: int
    total_model_calls: int
    successful_model_calls: int
    failed_model_calls: int
    total_case_elapsed_seconds: float
    average_case_elapsed_seconds: float
    usage_reported_calls: int
    usage_unavailable_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    decision_accuracy: float
    tool_name_accuracy: float | None
    arguments_accuracy: float | None
    case_accuracy: float
    planner_identity: str
    results: tuple[JobAgentPlannerCaseResult, ...]


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationPreflightReport:
    total_cases: int
    call_tool_cases: int
    refusal_cases: int
    registered_tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationProgress:
    completed_cases: int
    total_cases: int
    case_id: str
    execution_status: Literal["completed", "planner_failed"]
    elapsed_seconds: float


def validate_job_agent_planner_evaluation_cases(
    cases: tuple[JobAgentPlannerEvaluationCase, ...],
    *,
    tools: tuple[AgentToolDefinition, ...],
    argument_normalizers: Mapping[
        str,
        AgentToolArgumentsNormalizer,
    ]
    | None = None,
) -> JobAgentPlannerEvaluationPreflightReport:
    if not cases:
        raise ValueError("planner evaluation cases must not be empty")
    if not tools:
        raise ValueError("planner evaluation requires tools")

    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("planner evaluation case IDs must be unique")

    registered_tool_names = {tool.name for tool in tools}
    normalizers = dict(argument_normalizers or {})
    unknown_normalizer_names = set(normalizers) - registered_tool_names
    if unknown_normalizer_names:
        raise ValueError(
            "argument normalizer references an unregistered tool"
        )

    call_tool_cases = 0
    for case in cases:
        if case.expected_decision != "call_tool":
            continue
        call_tool_cases += 1
        if case.expected_tool_name not in registered_tool_names:
            raise ValueError(
                "evaluation case references an unregistered tool: "
                f"{case.case_id}"
            )
        if case.expected_arguments is None:
            raise AssertionError("validated call_tool case requires arguments")
        normalizer = normalizers.get(case.expected_tool_name)
        if normalizer is None:
            continue
        try:
            normalizer.normalize(case.expected_arguments)
        except AgentToolArgumentsNormalizationError as error:
            raise ValueError(
                "invalid expected arguments for evaluation case: "
                f"{case.case_id}"
            ) from error

    return JobAgentPlannerEvaluationPreflightReport(
        total_cases=len(cases),
        call_tool_cases=call_tool_cases,
        refusal_cases=len(cases) - call_tool_cases,
        registered_tool_names=tuple(sorted(registered_tool_names)),
    )


class JobAgentPlannerEvaluator:
    def __init__(
        self,
        planner: JobAgentPlanner,
        *,
        tools: tuple[AgentToolDefinition, ...],
        argument_normalizers: Mapping[
            str,
            AgentToolArgumentsNormalizer,
        ]
        | None = None,
        monotonic_clock: Callable[[], float] = perf_counter,
        progress_observer: Callable[
            [JobAgentPlannerEvaluationProgress],
            None,
        ]
        | None = None,
    ) -> None:
        if not tools:
            raise ValueError("planner evaluation requires tools")
        registered_tool_names = {tool.name for tool in tools}
        normalizers = dict(argument_normalizers or {})
        unknown_normalizer_names = set(normalizers) - registered_tool_names
        if unknown_normalizer_names:
            raise ValueError(
                "argument normalizer references an unregistered tool"
            )
        self._planner = planner
        self._tools = tools
        self._argument_normalizers = normalizers
        self._monotonic_clock = monotonic_clock
        self._progress_observer = progress_observer

    def evaluate(
        self,
        cases: tuple[JobAgentPlannerEvaluationCase, ...],
    ) -> JobAgentPlannerEvaluationReport:
        self.preflight(cases)

        collected_results: list[JobAgentPlannerCaseResult] = []
        for completed_cases, case in enumerate(cases, start=1):
            result = self._evaluate_case(case)
            collected_results.append(result)
            if self._progress_observer is not None:
                self._progress_observer(
                    JobAgentPlannerEvaluationProgress(
                        completed_cases=completed_cases,
                        total_cases=len(cases),
                        case_id=case.case_id,
                        execution_status=result.execution_status,
                        elapsed_seconds=result.elapsed_seconds,
                    )
                )
        results = tuple(collected_results)
        failed_model_calls = sum(
            result.execution_status == "planner_failed"
            for result in results
        )
        total_case_elapsed_seconds = sum(
            result.elapsed_seconds for result in results
        )
        usage_reported_calls = sum(
            result.total_tokens is not None for result in results
        )
        tool_results = tuple(
            result
            for case, result in zip(cases, results, strict=True)
            if case.expected_decision == "call_tool"
        )
        return JobAgentPlannerEvaluationReport(
            total_cases=len(cases),
            total_model_calls=len(cases),
            successful_model_calls=len(cases) - failed_model_calls,
            failed_model_calls=failed_model_calls,
            total_case_elapsed_seconds=total_case_elapsed_seconds,
            average_case_elapsed_seconds=(
                total_case_elapsed_seconds / len(cases)
            ),
            usage_reported_calls=usage_reported_calls,
            usage_unavailable_calls=len(cases) - usage_reported_calls,
            total_input_tokens=sum(result.input_tokens or 0 for result in results),
            total_output_tokens=sum(result.output_tokens or 0 for result in results),
            total_tokens=sum(result.total_tokens or 0 for result in results),
            decision_accuracy=(
                sum(result.decision_correct for result in results)
                / len(results)
            ),
            tool_name_accuracy=(
                sum(result.tool_name_correct is True for result in tool_results)
                / len(tool_results)
                if tool_results
                else None
            ),
            arguments_accuracy=(
                sum(result.arguments_correct is True for result in tool_results)
                / len(tool_results)
                if tool_results
                else None
            ),
            case_accuracy=(
                sum(result.passed for result in results) / len(results)
            ),
            planner_identity=self._planner.identity,
            results=results,
        )

    def preflight(
        self,
        cases: tuple[JobAgentPlannerEvaluationCase, ...],
    ) -> JobAgentPlannerEvaluationPreflightReport:
        return validate_job_agent_planner_evaluation_cases(
            cases,
            tools=self._tools,
            argument_normalizers=self._argument_normalizers,
        )

    def _evaluate_case(
        self,
        case: JobAgentPlannerEvaluationCase,
    ) -> JobAgentPlannerCaseResult:
        started_at = self._monotonic_clock()
        token_usage: JobAgentPlannerTokenUsage | None = None
        try:
            if isinstance(self._planner, ObservableJobAgentPlanner):
                observed = self._planner.plan_observed(
                    user_query=case.user_query,
                    tools=self._tools,
                )
                plan = observed.plan
                token_usage = observed.token_usage
            else:
                plan = self._planner.plan(
                    user_query=case.user_query,
                    tools=self._tools,
                )
        except Exception as error:
            return self._planner_failed_result(
                case,
                error,
                elapsed_seconds=max(
                    0.0,
                    self._monotonic_clock() - started_at,
                ),
            )
        actual_call = plan.tool_call
        actual_tool_name = actual_call.name if actual_call else None
        actual_arguments = actual_call.arguments if actual_call else None
        decision_correct = plan.decision == case.expected_decision

        if case.expected_decision == "refuse":
            tool_name_correct = None
            arguments_correct = None
            arguments_error_code = None
            passed = decision_correct and actual_call is None
        else:
            tool_name_correct = (
                actual_tool_name == case.expected_tool_name
            )
            arguments_correct, arguments_error_code = (
                self._compare_arguments(
                    case=case,
                    tool_name_correct=tool_name_correct,
                    actual_arguments=actual_arguments,
                )
            )
            passed = (
                decision_correct
                and tool_name_correct
                and arguments_correct
            )
        return JobAgentPlannerCaseResult(
            case_id=case.case_id,
            execution_status="completed",
            decision_correct=decision_correct,
            tool_name_correct=tool_name_correct,
            arguments_correct=arguments_correct,
            passed=passed,
            actual_decision=plan.decision,
            actual_tool_name=actual_tool_name,
            actual_arguments=actual_arguments,
            arguments_error_code=arguments_error_code,
            planner_error_code=None,
            planner_error_type=None,
            elapsed_seconds=max(
                0.0,
                self._monotonic_clock() - started_at,
            ),
            input_tokens=(
                token_usage.input_tokens if token_usage is not None else None
            ),
            output_tokens=(
                token_usage.output_tokens if token_usage is not None else None
            ),
            total_tokens=(
                token_usage.total_tokens if token_usage is not None else None
            ),
        )

    @staticmethod
    def _planner_failed_result(
        case: JobAgentPlannerEvaluationCase,
        error: Exception,
        *,
        elapsed_seconds: float,
    ) -> JobAgentPlannerCaseResult:
        expects_tool = case.expected_decision == "call_tool"
        return JobAgentPlannerCaseResult(
            case_id=case.case_id,
            execution_status="planner_failed",
            decision_correct=False,
            tool_name_correct=False if expects_tool else None,
            arguments_correct=False if expects_tool else None,
            passed=False,
            actual_decision=None,
            actual_tool_name=None,
            actual_arguments=None,
            arguments_error_code=None,
            planner_error_code="planner-call-failed",
            planner_error_type=type(error).__name__,
            elapsed_seconds=elapsed_seconds,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )

    def _compare_arguments(
        self,
        *,
        case: JobAgentPlannerEvaluationCase,
        tool_name_correct: bool,
        actual_arguments: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        expected_tool_name = case.expected_tool_name
        expected_arguments = case.expected_arguments
        if (
            not tool_name_correct
            or expected_tool_name is None
            or expected_arguments is None
            or actual_arguments is None
        ):
            return False, None

        normalizer = self._argument_normalizers.get(expected_tool_name)
        if normalizer is None:
            return actual_arguments == expected_arguments, None
        expected_normalized = normalizer.normalize(expected_arguments)
        try:
            actual_normalized = normalizer.normalize(actual_arguments)
        except AgentToolArgumentsNormalizationError:
            return False, "actual-arguments-invalid"
        return actual_normalized == expected_normalized, None
