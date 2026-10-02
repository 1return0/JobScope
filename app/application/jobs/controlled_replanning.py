"""Fixed, auditable limits for a single read-only Agent replanning attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ReplanDecision = Literal["continue", "replan_once", "stop"]


@dataclass(frozen=True, slots=True)
class ReplanAssessment:
    decision: ReplanDecision
    reason_code: str


@dataclass(frozen=True, slots=True)
class ReplanAuditRecord:
    """Immutable explanation of a bounded recovery decision."""

    original_query: str
    original_constraints: tuple[tuple[str, str], ...]
    result_code: str
    replan_count: int
    assessment: ReplanAssessment

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError("original_query must not be blank")
        if self.replan_count < 0:
            raise ValueError("replan_count must not be negative")


class ControlledReplanPolicy:
    """Allows at most one recovery attempt for safe read-only evidence gaps."""

    _RECOVERABLE_READ_FAILURES = frozenset(
        {"insufficient_evidence", "no_matches"}
    )

    def assess(
        self,
        *,
        result_code: str,
        replan_count: int,
        operation_kind: Literal["read", "write"],
    ) -> ReplanAssessment:
        if replan_count < 0:
            raise ValueError("replan_count must not be negative")
        if operation_kind == "write":
            return ReplanAssessment("stop", "write-operation-never-replanned")
        if replan_count >= 1:
            return ReplanAssessment("stop", "replan-limit-reached")
        if result_code in self._RECOVERABLE_READ_FAILURES:
            return ReplanAssessment("replan_once", result_code)
        return ReplanAssessment("stop", "non-recoverable-result")

    def assess_with_audit(
        self,
        *,
        original_query: str,
        original_constraints: dict[str, str],
        result_code: str,
        replan_count: int,
        operation_kind: Literal["read", "write"],
    ) -> ReplanAuditRecord:
        assessment = self.assess(
            result_code=result_code,
            replan_count=replan_count,
            operation_kind=operation_kind,
        )
        return ReplanAuditRecord(
            original_query=original_query,
            original_constraints=tuple(sorted(original_constraints.items())),
            result_code=result_code,
            replan_count=replan_count,
            assessment=assessment,
        )
