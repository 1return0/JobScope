from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerEvaluator,
    JobAgentPlannerEvaluationProgress,
    PydanticAgentToolArgumentsNormalizer,
    validate_job_agent_planner_evaluation_cases,
)
from app.application.jobs.job_agent_planner_evaluation_json import (
    load_job_agent_planner_evaluation_dataset,
)
from app.application.jobs.job_agent_planner_evaluation_gate import (
    JobAgentPlannerEvaluationGatePolicy,
    evaluate_job_agent_planner_report_gate,
)
from app.application.jobs.job_search_tool import (
    SearchCurrentJobsArguments,
    SearchCurrentJobsTool,
)
from app.application.jobs.job_agent_planning import AgentToolDefinition
from app.config import PROJECT_ROOT, load_settings
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    OpenAiCompatibleJobAgentPlannerConfig,
    build_openai_compatible_job_agent_planner,
)
from app.infrastructure.reports.job_agent_planner_experiment_report_json import (
    save_job_agent_planner_experiment_report,
    serialize_job_agent_planner_experiment_report,
)


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "job-agent-planner-v1.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    dataset = load_job_agent_planner_evaluation_dataset(arguments.dataset)
    tools = (_search_tool_definition(),)
    argument_normalizers = {
        SearchCurrentJobsTool.name: PydanticAgentToolArgumentsNormalizer(
            SearchCurrentJobsArguments
        )
    }
    if arguments.validate_only:
        preflight = validate_job_agent_planner_evaluation_cases(
            dataset.cases,
            tools=tools,
            argument_normalizers=argument_normalizers,
        )
        payload = {
            "status": "validated",
            "model_calls": 0,
            "dataset_id": dataset.dataset_id,
            "dataset_version": dataset.dataset_version,
            "dataset_sha256": dataset.dataset_sha256,
            "total_cases": preflight.total_cases,
            "call_tool_cases": preflight.call_tool_cases,
            "refusal_cases": preflight.refusal_cases,
            "registered_tool_names": preflight.registered_tool_names,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    settings = load_settings(dotenv_override=arguments.dotenv_override)
    planner = build_openai_compatible_job_agent_planner(
        OpenAiCompatibleJobAgentPlannerConfig(
            model_name=settings.answer_model_name,
            base_url=settings.answer_model_base_url,
            api_key_environment_variable=(
                settings.answer_model_api_key_environment_variable
            ),
            timeout_seconds=settings.answer_model_timeout_seconds,
            max_completion_tokens=min(
                settings.answer_model_max_completion_tokens,
                800,
            ),
            enable_thinking=False,
        )
    )
    report = JobAgentPlannerEvaluator(
        planner,
        tools=tools,
        argument_normalizers=argument_normalizers,
        progress_observer=_print_progress,
    ).evaluate(dataset.cases)
    gate_result = (
        evaluate_job_agent_planner_report_gate(
            report,
            JobAgentPlannerEvaluationGatePolicy(
                minimum_case_accuracy=arguments.minimum_case_accuracy,
                maximum_failed_model_calls=arguments.maximum_failed_model_calls,
                maximum_average_case_elapsed_seconds=(
                    arguments.maximum_average_case_elapsed_seconds
                ),
                maximum_usage_unavailable_calls=(
                    arguments.maximum_usage_unavailable_calls
                ),
                maximum_average_tokens_per_reported_call=(
                    arguments.maximum_average_tokens_per_reported_call
                ),
            ),
        )
        if arguments.enforce_gate
        else None
    )
    payload = serialize_job_agent_planner_experiment_report(
        dataset,
        report,
        executed_at=datetime.now(timezone.utc),
        gate_result=gate_result,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if arguments.output is not None:
        save_job_agent_planner_experiment_report(arguments.output, payload)
    return 0 if gate_result is None or gate_result.passed else 2


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate real Job Agent planner decisions."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset and golden arguments without model calls.",
    )
    parser.add_argument("--enforce-gate", action="store_true")
    parser.add_argument("--minimum-case-accuracy", type=float, default=0.8)
    parser.add_argument("--maximum-failed-model-calls", type=int, default=0)
    parser.add_argument(
        "--maximum-average-case-elapsed-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--maximum-usage-unavailable-calls",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--maximum-average-tokens-per-reported-call",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--dotenv-override",
        action="store_true",
        help=(
            "Let project .env replace inherited process variables for this "
            "evaluation process only."
        ),
    )
    return parser.parse_args(argv)


def _search_tool_definition() -> AgentToolDefinition:
    return AgentToolDefinition(
        name=SearchCurrentJobsTool.name,
        description=SearchCurrentJobsTool.description,
        arguments_schema=SearchCurrentJobsArguments.model_json_schema(),
    )


def _print_progress(progress: JobAgentPlannerEvaluationProgress) -> None:
    print(
        (
            f"[{progress.completed_cases}/{progress.total_cases}] "
            f"{progress.case_id} {progress.execution_status} "
            f"{progress.elapsed_seconds:.3f}s"
        ),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
