from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.application.answering.grounded_answer_evaluation import (
    GroundedAnswerEvaluationReport,
    GroundedAnswerEvaluator,
)
from app.application.answering.grounded_answer_evaluation_json import (
    GroundedAnswerEvaluationDataset,
    load_grounded_answer_evaluation_dataset,
)
from app.application.answering.grounded_answer_service import GroundedAnswerService
from app.application.answering.answer_prompting import (
    GROUNDED_ANSWER_PROMPT_VERSION,
)
from app.config import PROJECT_ROOT, load_settings
from app.infrastructure.llm.openai_compatible_answer_model import (
    OpenAiCompatibleAnswerModelConfig,
    build_openai_compatible_answer_model,
)


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "grounded-answer-smoke-v1.json"
)


def main() -> int:
    arguments = _parse_arguments()
    settings = load_settings(dotenv_override=True)
    if not settings.answer_generation_enabled:
        raise RuntimeError(
            "set JOBSCOPE_ANSWER_GENERATION_ENABLED=true before evaluation"
        )
    dataset = load_grounded_answer_evaluation_dataset(arguments.dataset)
    model = build_openai_compatible_answer_model(
        OpenAiCompatibleAnswerModelConfig(
            model_name=settings.answer_model_name,
            base_url=settings.answer_model_base_url,
            api_key_environment_variable=(
                settings.answer_model_api_key_environment_variable
            ),
            timeout_seconds=settings.answer_model_timeout_seconds,
            max_completion_tokens=(
                settings.answer_model_max_completion_tokens
            ),
        )
    )
    report = GroundedAnswerEvaluator(
        GroundedAnswerService(model)
    ).evaluate(dataset.cases)
    payload = serialize_grounded_answer_evaluation_report(
        dataset,
        report,
        model_name=settings.answer_model_name,
        dataset_sha256=_sha256(arguments.dataset),
        prompt_version=GROUNDED_ANSWER_PROMPT_VERSION,
        executed_at=datetime.now(timezone.utc),
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if arguments.output is not None:
        save_report(arguments.output, rendered)
    return 0


def serialize_grounded_answer_evaluation_report(
    dataset: GroundedAnswerEvaluationDataset,
    report: GroundedAnswerEvaluationReport,
    *,
    model_name: str,
    dataset_sha256: str,
    prompt_version: str,
    executed_at: datetime,
) -> dict[str, object]:
    if executed_at.tzinfo is None:
        raise ValueError("answer evaluation executed_at must have timezone")
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "review_status": dataset.review_status,
        "model_name": model_name,
        "dataset_sha256": dataset_sha256,
        "prompt_version": prompt_version,
        "executed_at": executed_at.isoformat(),
        "case_count": report.case_count,
        "metrics": {
            "status_accuracy": report.status_accuracy,
            "answer_accuracy": report.answer_accuracy,
            "refusal_accuracy": report.refusal_accuracy,
            "grounding_pass_rate": report.grounding_pass_rate,
            "model_call_rate": report.model_call_rate,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "actual_status": case.actual_status,
                "status_correct": case.status_correct,
                "grounding_valid": case.grounding_valid,
                "model_called": case.model_called,
                "failure_code": case.failure_code,
            }
            for case in report.cases
        ],
        "limitations": [
            "provisional cases are not resume-grade metrics",
            "retrieved evidence is frozen and does not test retrieval",
            "status and string grounding do not prove semantic correctness",
        ],
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded answer and refusal behavior."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def save_report(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as report_file:
        report_file.write(rendered)
        report_file.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
