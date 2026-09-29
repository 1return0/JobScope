from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AnswerExperimentAxis = Literal[
    "repeatability",
    "model",
    "prompt",
]


@dataclass(frozen=True, slots=True)
class GroundedAnswerExperimentIdentity:
    dataset_id: str
    dataset_version: int
    dataset_sha256: str
    model_name: str
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be blank")
        if self.dataset_version < 1:
            raise ValueError("dataset_version must be positive")
        if len(self.dataset_sha256) != 64:
            raise ValueError("dataset_sha256 must contain 64 characters")
        if not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must not be blank")


@dataclass(frozen=True, slots=True)
class GroundedAnswerExperimentComparison:
    axis: AnswerExperimentAxis
    changed_fields: tuple[str, ...]


def classify_grounded_answer_experiment(
    baseline: GroundedAnswerExperimentIdentity,
    candidate: GroundedAnswerExperimentIdentity,
) -> GroundedAnswerExperimentComparison:
    if (
        baseline.dataset_id != candidate.dataset_id
        or baseline.dataset_version != candidate.dataset_version
        or baseline.dataset_sha256 != candidate.dataset_sha256
    ):
        raise ValueError(
            "answer experiments use different evaluation datasets"
        )

    changed_fields = tuple(
        field_name
        for field_name in ("model_name", "prompt_version")
        if getattr(baseline, field_name) != getattr(candidate, field_name)
    )
    if len(changed_fields) > 1:
        raise ValueError(
            "answer experiment changes more than one independent variable"
        )
    if not changed_fields:
        return GroundedAnswerExperimentComparison(
            axis="repeatability",
            changed_fields=(),
        )
    return GroundedAnswerExperimentComparison(
        axis=("model" if changed_fields[0] == "model_name" else "prompt"),
        changed_fields=changed_fields,
    )
