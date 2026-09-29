from __future__ import annotations

from dataclasses import dataclass

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerEvaluationReport,
)


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationGatePolicy:
    minimum_case_accuracy: float = 0.8
    maximum_failed_model_calls: int = 0
    maximum_average_case_elapsed_seconds: float = 5.0
    maximum_usage_unavailable_calls: int = 0
    maximum_average_tokens_per_reported_call: float = 1000.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_case_accuracy <= 1.0:
            raise ValueError("minimum_case_accuracy must be between 0 and 1")
        if self.maximum_failed_model_calls < 0:
            raise ValueError("maximum_failed_model_calls must not be negative")
        if self.maximum_average_case_elapsed_seconds <= 0:
            raise ValueError(
                "maximum_average_case_elapsed_seconds must be positive"
            )
        if self.maximum_usage_unavailable_calls < 0:
            raise ValueError(
                "maximum_usage_unavailable_calls must not be negative"
            )
        if self.maximum_average_tokens_per_reported_call <= 0:
            raise ValueError(
                "maximum_average_tokens_per_reported_call must be positive"
            )


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationGateViolation:
    code: str
    actual: float
    threshold: float


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationGateResult:
    passed: bool
    average_tokens_per_reported_call: float | None
    violations: tuple[JobAgentPlannerEvaluationGateViolation, ...]


def evaluate_job_agent_planner_report_gate(
    report: JobAgentPlannerEvaluationReport,
    policy: JobAgentPlannerEvaluationGatePolicy,
) -> JobAgentPlannerEvaluationGateResult:
    violations: list[JobAgentPlannerEvaluationGateViolation] = []

    def reject_if(
        condition: bool,
        *,
        code: str,
        actual: float,
        threshold: float,
    ) -> None:
        if condition:
            violations.append(
                JobAgentPlannerEvaluationGateViolation(
                    code=code,
                    actual=actual,
                    threshold=threshold,
                )
            )

    reject_if(
        report.case_accuracy < policy.minimum_case_accuracy,
        code="case-accuracy-below-minimum",
        actual=report.case_accuracy,
        threshold=policy.minimum_case_accuracy,
    )
    reject_if(
        report.failed_model_calls > policy.maximum_failed_model_calls,
        code="model-failures-above-maximum",
        actual=float(report.failed_model_calls),
        threshold=float(policy.maximum_failed_model_calls),
    )
    reject_if(
        report.average_case_elapsed_seconds
        > policy.maximum_average_case_elapsed_seconds,
        code="average-latency-above-maximum",
        actual=report.average_case_elapsed_seconds,
        threshold=policy.maximum_average_case_elapsed_seconds,
    )
    reject_if(
        report.usage_unavailable_calls
        > policy.maximum_usage_unavailable_calls,
        code="usage-unavailable-above-maximum",
        actual=float(report.usage_unavailable_calls),
        threshold=float(policy.maximum_usage_unavailable_calls),
    )

    average_tokens = (
        report.total_tokens / report.usage_reported_calls
        if report.usage_reported_calls > 0
        else None
    )
    if average_tokens is not None:
        reject_if(
            average_tokens
            > policy.maximum_average_tokens_per_reported_call,
            code="average-tokens-above-maximum",
            actual=average_tokens,
            threshold=policy.maximum_average_tokens_per_reported_call,
        )

    return JobAgentPlannerEvaluationGateResult(
        passed=not violations,
        average_tokens_per_reported_call=average_tokens,
        violations=tuple(violations),
    )
