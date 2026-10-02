from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

from app.application.jobs.job_agent_sessions import JobAgentSessionRecord
from app.infrastructure.persistence.models import JobAgentSessionRow


class SqlAlchemyJobAgentSessionStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def create(self, record: JobAgentSessionRecord) -> None:
        with self._session_factory.begin() as session:
            session.add(
                JobAgentSessionRow(
                    token_hash=record.token_hash,
                    owner_id=record.owner_id,
                    expires_at=record.expires_at,
                    created_at=record.created_at,
                    last_seen_at=record.last_seen_at,
                )
            )

    def find_active(
        self,
        *,
        token_hash: str,
        now: datetime,
    ) -> JobAgentSessionRecord | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(JobAgentSessionRow).where(
                    JobAgentSessionRow.token_hash == token_hash,
                    JobAgentSessionRow.expires_at > now,
                )
            )
            return _to_record(row) if row is not None else None

    def mark_seen(self, *, token_hash: str, seen_at: datetime) -> None:
        with self._session_factory.begin() as session:
            session.execute(
                update(JobAgentSessionRow)
                .where(JobAgentSessionRow.token_hash == token_hash)
                .values(last_seen_at=seen_at)
            )


def _to_record(row: JobAgentSessionRow) -> JobAgentSessionRecord:
    return JobAgentSessionRecord(
        token_hash=row.token_hash,
        owner_id=row.owner_id,
        expires_at=_as_utc(row.expires_at),
        created_at=_as_utc(row.created_at),
        last_seen_at=_as_utc(row.last_seen_at),
    )


def _as_utc(value: datetime) -> datetime:
    """SQLite test storage drops tzinfo; PostgreSQL preserves it."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
