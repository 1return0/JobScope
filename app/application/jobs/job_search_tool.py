from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.application.jobs.current_job_records import (
    CurrentJobRecordFilters,
    CurrentJobRecordReader,
    CurrentStructuredJobRecord,
)
from app.domain.jobs.recruitment_type import (
    RecruitmentType,
    normalize_recruitment_type,
)


class SearchCurrentJobsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str | None = Field(default=None, min_length=1, max_length=500)
    job_title: str | None = Field(default=None, min_length=1, max_length=500)
    location: str | None = Field(default=None, min_length=1, max_length=200)
    recruitment_type: RecruitmentType | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    deadline_on_or_after: date | None = None
    deadline_on_or_before: date | None = None
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("recruitment_type", mode="before")
    @classmethod
    def normalize_recruitment_type_alias(
        cls,
        value: object,
    ) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return normalize_recruitment_type(value)

    @model_validator(mode="after")
    def validate_deadline_range(self) -> SearchCurrentJobsArguments:
        if (
            self.deadline_on_or_after is not None
            and self.deadline_on_or_before is not None
            and self.deadline_on_or_after > self.deadline_on_or_before
        ):
            raise ValueError("deadline range is reversed")
        return self


class CurrentJobToolResult(BaseModel):
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


class SearchCurrentJobsOutput(BaseModel):
    status: Literal["found", "no_matches"]
    result_count: int
    jobs: tuple[CurrentJobToolResult, ...]


class SearchCurrentJobsTool:
    name = "search_current_jobs"
    description = (
        "Search only activated job records whose source document snapshot "
        "is still current. Supports company, job title, location, "
        "recruitment type and normalized application-deadline filters."
    )

    def __init__(self, reader: CurrentJobRecordReader) -> None:
        self._reader = reader

    @property
    def arguments_schema(self) -> dict[str, Any]:
        return SearchCurrentJobsArguments.model_json_schema()

    def execute(
        self,
        arguments: SearchCurrentJobsArguments,
    ) -> SearchCurrentJobsOutput:
        rows = self._reader.list_current(
            CurrentJobRecordFilters(
                company=arguments.company,
                job_title=arguments.job_title,
                location=arguments.location,
                recruitment_type=arguments.recruitment_type,
                deadline_on_or_after=arguments.deadline_on_or_after,
                deadline_on_or_before=arguments.deadline_on_or_before,
                limit=arguments.limit,
            )
        )
        jobs = tuple(self._to_result(row) for row in rows)
        return SearchCurrentJobsOutput(
            status="found" if jobs else "no_matches",
            result_count=len(jobs),
            jobs=jobs,
        )

    def invoke(self, raw_arguments: dict[str, Any]) -> SearchCurrentJobsOutput:
        return self.execute(
            SearchCurrentJobsArguments.model_validate(raw_arguments)
        )

    @staticmethod
    def _to_result(
        row: CurrentStructuredJobRecord,
    ) -> CurrentJobToolResult:
        return CurrentJobToolResult(
            record_id=row.record_id,
            source_snapshot_id=row.source_snapshot_id,
            company=row.company,
            job_title=row.job_title,
            locations=row.locations,
            education_requirement=row.education_requirement,
            major_requirement=row.major_requirement,
            recruitment_type=row.recruitment_type,
            application_deadline_raw=row.application_deadline_raw,
            application_deadline_normalized=(
                row.application_deadline_normalized
            ),
            deadline_normalization_status=(
                row.deadline_normalization_status
            ),
            source_reference=row.source_reference,
        )
