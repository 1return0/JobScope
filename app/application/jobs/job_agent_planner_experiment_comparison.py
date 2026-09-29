from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerEvaluationReport,
)


ComparisonKind = Literal["regression", "controlled_experiment"]


@dataclass(frozen=True, slots=True)
class JobAgentPlannerExperimentRun:
    run_id: str
    dataset_sha256: str
    report: JobAgentPlannerEvaluationReport

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("experiment run_id must not be blank")
        if len(self.dataset_sha256) != 64:
            raise ValueError("experiment dataset_sha256 must have 64 characters")


@dataclass(frozen=True, slots=True)
class JobAgentPlannerExperimentComparison:
    comparison_kind: ComparisonKind
    changed_factor: str | None
    baseline_run_id: str
    candidate_run_id: str
    case_accuracy_delta: float
    decision_accuracy_delta: float
    average_latency_seconds_delta: float
    average_tokens_delta: float | None
    failed_model_calls_delta: int
    regressed_case_ids: tuple[str, ...]
    improved_case_ids: tuple[str, ...]


def compare_job_agent_planner_experiments(
    baseline: JobAgentPlannerExperimentRun,
    candidate: JobAgentPlannerExperimentRun,
    *,
    comparison_kind: ComparisonKind = "regression",
    changed_factor: str | None = None,
) -> JobAgentPlannerExperimentComparison:
    if baseline.dataset_sha256 != candidate.dataset_sha256:
        raise ValueError("experiment comparison requires identical dataset_sha256")

    baseline_results = {result.case_id: result for result in baseline.report.results}
    candidate_results = {result.case_id: result for result in candidate.report.results}
    if set(baseline_results) != set(candidate_results):
        raise ValueError("experiment comparison requires identical case IDs")

    if comparison_kind == "regression":
        if baseline.report.planner_identity != candidate.report.planner_identity:
            raise ValueError(
                "regression comparison requires identical planner identity"
            )
        if changed_factor is not None:
            raise ValueError("regression comparison must not declare changed_factor")
        normalized_factor = None
    else:
        normalized_factor = (changed_factor or "").strip()
        if not normalized_factor:
            raise ValueError(
                "controlled experiment requires one declared changed_factor"
            )

    return JobAgentPlannerExperimentComparison(
        comparison_kind=comparison_kind,
        changed_factor=normalized_factor,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        case_accuracy_delta=(
            candidate.report.case_accuracy - baseline.report.case_accuracy
        ),
        decision_accuracy_delta=(
            candidate.report.decision_accuracy
            - baseline.report.decision_accuracy
        ),
        average_latency_seconds_delta=(
            candidate.report.average_case_elapsed_seconds
            - baseline.report.average_case_elapsed_seconds
        ),
        average_tokens_delta=_average_tokens_delta(baseline, candidate),
        failed_model_calls_delta=(
            candidate.report.failed_model_calls
            - baseline.report.failed_model_calls
        ),
        regressed_case_ids=tuple(
            sorted(
                case_id
                for case_id, baseline_result in baseline_results.items()
                if baseline_result.passed
                and not candidate_results[case_id].passed
            )
        ),
        improved_case_ids=tuple(
            sorted(
                case_id
                for case_id, baseline_result in baseline_results.items()
                if not baseline_result.passed
                and candidate_results[case_id].passed
            )
        ),
    )


def _average_tokens_delta(
    baseline: JobAgentPlannerExperimentRun,
    candidate: JobAgentPlannerExperimentRun,
) -> float | None:
    reports = (baseline.report, candidate.report)
    if any(
        report.usage_unavailable_calls > 0
        or report.usage_reported_calls == 0
        for report in reports
    ):
        return None
    baseline_average = (
        baseline.report.total_tokens / baseline.report.usage_reported_calls
    )
    candidate_average = (
        candidate.report.total_tokens / candidate.report.usage_reported_calls
    )
    return candidate_average - baseline_average
