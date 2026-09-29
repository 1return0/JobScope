from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from app.domain.jobs.recruitment_type import (
    RecruitmentType,
    normalize_recruitment_type,
)


@dataclass(frozen=True, slots=True)
class CurrentJobRecordFilters:
    company: str | None = None
    job_title: str | None = None
    location: str | None = None
    recruitment_type: RecruitmentType | None = None
    deadline_on_or_after: date | None = None
    deadline_on_or_before: date | None = None
    limit: int = 50

    def __post_init__(self) -> None:
        for field_name in (
            "company",
            "job_title",
            "location",
            "recruitment_type",
        ):
            value = getattr(self, field_name)
            if value is not None:
                normalized = value.strip()
                if not normalized:
                    raise ValueError(f"{field_name} must not be blank")
                object.__setattr__(self, field_name, normalized)
        if self.recruitment_type is not None:
            object.__setattr__(
                self,
                "recruitment_type",
                normalize_recruitment_type(self.recruitment_type),
            )
        if not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if (
            self.deadline_on_or_after is not None
            and self.deadline_on_or_before is not None
            and self.deadline_on_or_after
            > self.deadline_on_or_before
        ):
            raise ValueError("deadline range is reversed")


@dataclass(frozen=True, slots=True)
class CurrentStructuredJobRecord:
    record_id: str
    source_snapshot_id: str
    company: str | None
    job_title: str | None
    locations: tuple[str, ...]
    education_requirement: str | None
    major_requirement: str | None
    recruitment_type: str | None
    application_deadline_raw: str | None
    application_deadline_normalized: date | None
    deadline_normalization_status: str
    source_reference: str | None = None


class CurrentJobRecordReader(Protocol):
    def list_current(
        self,
        filters: CurrentJobRecordFilters | None = None,
    ) -> list[CurrentStructuredJobRecord]:
        ...
