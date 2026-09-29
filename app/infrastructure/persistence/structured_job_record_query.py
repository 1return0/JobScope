from __future__ import annotations

from sqlalchemy import case, func, select
from sqlalchemy.orm import sessionmaker

from app.application.jobs.current_job_records import (
    CurrentJobRecordFilters,
    CurrentStructuredJobRecord,
)
from app.infrastructure.persistence.models import (
    DocumentSnapshotRow,
    StructuredJobRecordRow,
)
from app.domain.jobs.recruitment_type import recruitment_type_storage_aliases


class SqlAlchemyCurrentJobRecordQuery:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def list_current(
        self,
        filters: CurrentJobRecordFilters | None = None,
    ) -> list[CurrentStructuredJobRecord]:
        selected_filters = filters or CurrentJobRecordFilters()
        statement = (
            select(
                StructuredJobRecordRow,
                DocumentSnapshotRow.source_reference,
            )
            .join(
                DocumentSnapshotRow,
                DocumentSnapshotRow.snapshot_id
                == StructuredJobRecordRow.source_snapshot_id,
            )
            .where(
                StructuredJobRecordRow.is_current.is_(True),
                DocumentSnapshotRow.is_current.is_(True),
            )
        )
        if selected_filters.company is not None:
            statement = statement.where(
                func.lower(StructuredJobRecordRow.company)
                == selected_filters.company.casefold()
            )
        if selected_filters.job_title is not None:
            statement = statement.where(
                func.lower(StructuredJobRecordRow.job_title).contains(
                    selected_filters.job_title.casefold()
                )
            )
        if selected_filters.recruitment_type is not None:
            statement = statement.where(
                func.lower(StructuredJobRecordRow.recruitment_type).in_(
                    tuple(
                        alias.casefold()
                        for alias in recruitment_type_storage_aliases(
                            selected_filters.recruitment_type
                        )
                    )
                )
            )
        if selected_filters.deadline_on_or_after is not None:
            statement = statement.where(
                StructuredJobRecordRow.deadline_normalization_status
                == "normalized",
                StructuredJobRecordRow.application_deadline_normalized
                >= selected_filters.deadline_on_or_after,
            )
        if selected_filters.deadline_on_or_before is not None:
            statement = statement.where(
                StructuredJobRecordRow.deadline_normalization_status
                == "normalized",
                StructuredJobRecordRow.application_deadline_normalized
                <= selected_filters.deadline_on_or_before,
            )
        statement = statement.order_by(
            case(
                (
                    StructuredJobRecordRow.application_deadline_normalized
                    .is_(None),
                    1,
                ),
                else_=0,
            ),
            StructuredJobRecordRow.application_deadline_normalized,
            StructuredJobRecordRow.company,
            StructuredJobRecordRow.job_title,
            StructuredJobRecordRow.record_id,
        )

        with self._session_factory() as session:
            rows = session.execute(statement).all()

        if selected_filters.location is not None:
            normalized_location = _location_match_key(
                selected_filters.location
            )
            rows = [
                row
                for row in rows
                if any(
                    _location_match_key(location) == normalized_location
                    for location in row[0].locations
                )
            ]
        return [
            self._to_domain(row, source_reference=source_reference)
            for row, source_reference
            in rows[: selected_filters.limit]
        ]

    @staticmethod
    def _to_domain(
        row: StructuredJobRecordRow,
        *,
        source_reference: str,
    ) -> CurrentStructuredJobRecord:
        return CurrentStructuredJobRecord(
            record_id=row.record_id,
            source_snapshot_id=row.source_snapshot_id,
            company=row.company,
            job_title=row.job_title,
            locations=tuple(row.locations),
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
            source_reference=source_reference,
        )


def _location_match_key(value: str) -> str:
    normalized = "".join(value.split()).casefold()
    if len(normalized) > 1 and normalized.endswith("市"):
        return normalized[:-1]
    return normalized
