from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    StructuredJobRecordDraft,
)


JobFactSupportLabel = Literal[
    "supported",
    "unsupported",
    "uncertain",
]


@dataclass(frozen=True, slots=True)
class JobFactSupportDecision:
    label: JobFactSupportLabel
    rationale: str

    def __post_init__(self) -> None:
        if self.label not in (
            "supported",
            "unsupported",
            "uncertain",
        ):
            raise ValueError("unsupported job fact support label")
        if not self.rationale.strip():
            raise ValueError("job fact support rationale must not be blank")


class JobFactSupportJudge(Protocol):
    @property
    def identity(self) -> str:
        ...

    def judge(
        self,
        *,
        field_path: str,
        fact: EvidenceBackedJobFact,
    ) -> JobFactSupportDecision:
        ...


@dataclass(frozen=True, slots=True)
class JobFactSupportEvaluation:
    field_path: str
    value: str
    decision: JobFactSupportDecision


@dataclass(frozen=True, slots=True)
class JobRecordSemanticSupportReport:
    valid: bool
    supported_count: int
    unsupported_count: int
    uncertain_count: int
    evaluations: tuple[JobFactSupportEvaluation, ...]
    judge_identity: str


class JobRecordSemanticSupportEvaluator:
    def __init__(self, judge: JobFactSupportJudge) -> None:
        if not judge.identity.strip():
            raise ValueError("job fact support judge identity must not be blank")
        self._judge = judge

    def evaluate(
        self,
        record: StructuredJobRecordDraft,
    ) -> JobRecordSemanticSupportReport:
        evaluations = tuple(
            JobFactSupportEvaluation(
                field_path=field_path,
                value=fact.value,
                decision=self._judge.judge(
                    field_path=field_path,
                    fact=fact,
                ),
            )
            for field_path, fact in _iter_stated_facts(record)
            if fact.value is not None
        )
        supported_count = sum(
            item.decision.label == "supported"
            for item in evaluations
        )
        unsupported_count = sum(
            item.decision.label == "unsupported"
            for item in evaluations
        )
        uncertain_count = sum(
            item.decision.label == "uncertain"
            for item in evaluations
        )
        return JobRecordSemanticSupportReport(
            valid=(
                unsupported_count == 0
                and uncertain_count == 0
            ),
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            uncertain_count=uncertain_count,
            evaluations=evaluations,
            judge_identity=self._judge.identity,
        )


def _iter_stated_facts(record: StructuredJobRecordDraft):
    for field_name in (
        "company",
        "job_title",
        "education_requirement",
        "major_requirement",
        "recruitment_type",
        "application_deadline",
    ):
        fact = getattr(record, field_name)
        if fact.is_stated:
            yield field_name, fact

    for field_name in (
        "locations",
        "responsibilities",
        "required_qualifications",
        "preferred_qualifications",
    ):
        for index, fact in enumerate(getattr(record, field_name)):
            if fact.is_stated:
                yield f"{field_name}[{index}]", fact
