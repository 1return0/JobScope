from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerEvaluationReport,
)
from app.application.jobs.job_agent_planner_evaluation_json import (
    JobAgentPlannerEvaluationDataset,
)
from app.application.jobs.job_agent_planner_evaluation_gate import (
    JobAgentPlannerEvaluationGateResult,
)


def serialize_job_agent_planner_experiment_report(
    dataset: JobAgentPlannerEvaluationDataset,
    report: JobAgentPlannerEvaluationReport,
    *,
    executed_at: datetime,
    gate_result: JobAgentPlannerEvaluationGateResult | None = None,
) -> dict[str, object]:
    if executed_at.tzinfo is None:
        raise ValueError("planner experiment executed_at must have timezone")

    payload: dict[str, object] = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "review_status": dataset.review_status,
        "dataset_sha256": dataset.dataset_sha256,
        "planner_identity": report.planner_identity,
        "executed_at": executed_at.isoformat(),
        "case_count": report.total_cases,
        "model_call_count": report.total_model_calls,
        "successful_model_call_count": report.successful_model_calls,
        "failed_model_call_count": report.failed_model_calls,
        "timing": {
            "total_case_elapsed_seconds": report.total_case_elapsed_seconds,
            "average_case_elapsed_seconds": report.average_case_elapsed_seconds,
        },
        "token_usage": {
            "reported_call_count": report.usage_reported_calls,
            "unavailable_call_count": report.usage_unavailable_calls,
            "input_tokens": report.total_input_tokens,
            "output_tokens": report.total_output_tokens,
            "total_tokens": report.total_tokens,
        },
        "metrics": {
            "decision_accuracy": report.decision_accuracy,
            "tool_name_accuracy": report.tool_name_accuracy,
            "arguments_accuracy": report.arguments_accuracy,
            "case_accuracy": report.case_accuracy,
        },
        "cases": [
            {
                "case_id": result.case_id,
                "execution_status": result.execution_status,
                "passed": result.passed,
                "decision_correct": result.decision_correct,
                "tool_name_correct": result.tool_name_correct,
                "arguments_correct": result.arguments_correct,
                "actual_decision": result.actual_decision,
                "actual_tool_name": result.actual_tool_name,
                "actual_arguments": result.actual_arguments,
                "arguments_error_code": result.arguments_error_code,
                "planner_error_code": result.planner_error_code,
                "planner_error_type": result.planner_error_type,
                "elapsed_seconds": result.elapsed_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
            }
            for result in report.results
        ],
        "limitations": [
            "provisional datasets are not resume-grade final metrics",
            "planner evaluation tests tool selection and arguments only",
            "tool execution, database results and final answers are not tested",
            "passing cases do not prove production traffic robustness",
        ],
    }
    if gate_result is not None:
        payload["quality_gate"] = {
            "passed": gate_result.passed,
            "average_tokens_per_reported_call": (
                gate_result.average_tokens_per_reported_call
            ),
            "violations": [
                {
                    "code": violation.code,
                    "actual": violation.actual,
                    "threshold": violation.threshold,
                }
                for violation in gate_result.violations
            ],
        }
    return payload


def save_job_agent_planner_experiment_report(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    with path.open("x", encoding="utf-8", newline="\n") as report_file:
        report_file.write(rendered)
        report_file.write("\n")
