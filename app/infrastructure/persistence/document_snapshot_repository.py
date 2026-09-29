from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.documents.document_ingestion import DocumentArtifact
from app.domain.documents.document_snapshot import (
    DocumentSnapshot,
    build_snapshot_id,
)
from app.infrastructure.persistence.models import DocumentSnapshotRow


@dataclass(frozen=True, slots=True)
class RecordSnapshotReport:
    snapshot: DocumentSnapshot
    created: bool
    content_changed: bool


class SqlAlchemyDocumentSnapshotRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def record(
        self,
        artifact: DocumentArtifact,
        *,
        observed_at: datetime,
    ) -> RecordSnapshotReport:
        with self._session_factory.begin() as session:
            return self.record_in_session(
                session,
                artifact,
                observed_at=observed_at,
            )

    def record_in_session(
        self,
        session: Session,
        artifact: DocumentArtifact,
        *,
        observed_at: datetime,
    ) -> RecordSnapshotReport:
        normalized_time = self._normalize_time(observed_at)
        source_reference = artifact.source_reference.strip()
        rows = session.scalars(
            select(DocumentSnapshotRow)
            .where(
                DocumentSnapshotRow.source_reference
                == source_reference
            )
            .order_by(
                DocumentSnapshotRow.version_number.desc()
            )
            .with_for_update()
        ).all()

        matching_row = next(
            (
                row
                for row in rows
                if row.content_sha256 == artifact.content_sha256
            ),
            None,
        )
        current_row = next(
            (row for row in rows if row.is_current),
            None,
        )

        if matching_row is not None:
            content_changed = (
                current_row is not None
                and current_row.snapshot_id
                != matching_row.snapshot_id
            )
            if content_changed:
                current_row.is_current = False
                matching_row.is_current = True
            if normalized_time > self._as_utc(
                matching_row.last_observed_at
            ):
                matching_row.last_observed_at = normalized_time
            snapshot = self._to_domain(matching_row)
            return RecordSnapshotReport(
                snapshot=snapshot,
                created=False,
                content_changed=content_changed,
            )

        if current_row is not None:
            current_row.is_current = False

        next_version = (
            max(
                (row.version_number for row in rows),
                default=0,
            )
            + 1
        )
        new_row = DocumentSnapshotRow(
            snapshot_id=build_snapshot_id(
                source_reference=source_reference,
                content_sha256=artifact.content_sha256,
            ),
            source_reference=source_reference,
            document_format=artifact.document_format,
            content_sha256=artifact.content_sha256,
            byte_size=artifact.byte_size,
            version_number=next_version,
            first_observed_at=normalized_time,
            last_observed_at=normalized_time,
            is_current=True,
        )
        session.add(new_row)
        session.flush()
        return RecordSnapshotReport(
            snapshot=self._to_domain(new_row),
            created=True,
            content_changed=current_row is not None,
        )

    def get_current(
        self,
        source_reference: str,
    ) -> DocumentSnapshot | None:
        statement = select(DocumentSnapshotRow).where(
            DocumentSnapshotRow.source_reference
            == source_reference.strip(),
            DocumentSnapshotRow.is_current.is_(True),
        )
        with self._session_factory() as session:
            row = session.scalar(statement)
        return None if row is None else self._to_domain(row)

    def list_history(
        self,
        source_reference: str,
    ) -> list[DocumentSnapshot]:
        statement: Select = (
            select(DocumentSnapshotRow)
            .where(
                DocumentSnapshotRow.source_reference
                == source_reference.strip()
            )
            .order_by(DocumentSnapshotRow.version_number)
        )
        with self._session_factory() as session:
            rows = session.scalars(statement).all()
        return [self._to_domain(row) for row in rows]

    @staticmethod
    def _normalize_time(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @classmethod
    def _to_domain(
        cls,
        row: DocumentSnapshotRow,
    ) -> DocumentSnapshot:
        return DocumentSnapshot(
            snapshot_id=row.snapshot_id,
            source_reference=row.source_reference,
            document_format=row.document_format,
            content_sha256=row.content_sha256,
            byte_size=row.byte_size,
            version_number=row.version_number,
            first_observed_at=cls._as_utc(row.first_observed_at),
            last_observed_at=cls._as_utc(row.last_observed_at),
            is_current=row.is_current,
        )
