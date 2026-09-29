from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.application.jobs.job_agent_planner_evaluation import (
    JobAgentPlannerEvaluationCase,
)


PlannerDatasetReviewStatus = Literal["provisional", "reviewed"]


class JobAgentPlannerEvaluationDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class JobAgentPlannerEvaluationDataset:
    dataset_id: str
    dataset_version: int
    review_status: PlannerDatasetReviewStatus
    cases: tuple[JobAgentPlannerEvaluationCase, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise JobAgentPlannerEvaluationDatasetError(
                "planner dataset_id must not be blank"
            )
        if self.dataset_version < 1:
            raise JobAgentPlannerEvaluationDatasetError(
                "planner dataset_version must be positive"
            )
        if self.review_status not in ("provisional", "reviewed"):
            raise JobAgentPlannerEvaluationDatasetError(
                "unsupported planner dataset review_status"
            )
        if not self.cases:
            raise JobAgentPlannerEvaluationDatasetError(
                "planner evaluation dataset must contain cases"
            )
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise JobAgentPlannerEvaluationDatasetError(
                "planner evaluation dataset case IDs must be unique"
            )

    @property
    def dataset_sha256(self) -> str:
        canonical = json.dumps(
            _dataset_payload(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def load_job_agent_planner_evaluation_dataset(
    path: Path,
) -> JobAgentPlannerEvaluationDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise JobAgentPlannerEvaluationDatasetError(
            "planner evaluation dataset could not be read"
        ) from error
    return decode_job_agent_planner_evaluation_dataset(payload)


def decode_job_agent_planner_evaluation_dataset(
    payload: object,
) -> JobAgentPlannerEvaluationDataset:
    try:
        if not isinstance(payload, dict):
            raise TypeError("planner dataset must be an object")
        _require_exact_keys(
            payload,
            {"dataset_id", "dataset_version", "review_status", "cases"},
        )
        return JobAgentPlannerEvaluationDataset(
            dataset_id=_string(payload, "dataset_id"),
            dataset_version=_integer(payload, "dataset_version"),
            review_status=_string(  # type: ignore[arg-type]
                payload,
                "review_status",
            ),
            cases=tuple(_decode_case(item) for item in _list(payload, "cases")),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, JobAgentPlannerEvaluationDatasetError):
            raise
        raise JobAgentPlannerEvaluationDatasetError(
            "planner evaluation dataset does not match the schema"
        ) from error


def _decode_case(payload: object) -> JobAgentPlannerEvaluationCase:
    if not isinstance(payload, dict):
        raise TypeError("planner evaluation case must be an object")
    _require_exact_keys(
        payload,
        {
            "case_id",
            "user_query",
            "expected_decision",
            "expected_tool_name",
            "expected_arguments",
        },
    )
    expected_arguments = payload["expected_arguments"]
    if expected_arguments is not None and not isinstance(
        expected_arguments,
        dict,
    ):
        raise TypeError("expected_arguments must be an object or null")
    if isinstance(expected_arguments, dict) and not all(
        isinstance(key, str) for key in expected_arguments
    ):
        raise TypeError("expected argument names must be strings")
    return JobAgentPlannerEvaluationCase(
        case_id=_string(payload, "case_id"),
        user_query=_string(payload, "user_query"),
        expected_decision=_string(  # type: ignore[arg-type]
            payload,
            "expected_decision",
        ),
        expected_tool_name=_optional_string(payload, "expected_tool_name"),
        expected_arguments=expected_arguments,  # type: ignore[arg-type]
    )


def _dataset_payload(
    dataset: JobAgentPlannerEvaluationDataset,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "review_status": dataset.review_status,
        "cases": [
            {
                "case_id": case.case_id,
                "user_query": case.user_query,
                "expected_decision": case.expected_decision,
                "expected_tool_name": case.expected_tool_name,
                "expected_arguments": case.expected_arguments,
            }
            for case in dataset.cases
        ],
    }


def _require_exact_keys(
    payload: dict[object, object],
    expected_keys: set[str],
) -> None:
    if set(payload) != expected_keys:
        raise TypeError("JSON object fields do not match the required schema")


def _string(payload: dict[object, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _optional_string(
    payload: dict[object, object],
    field: str,
) -> str | None:
    value = payload[field]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    return value


def _integer(payload: dict[object, object], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _list(payload: dict[object, object], field: str) -> list[object]:
    value = payload[field]
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value
