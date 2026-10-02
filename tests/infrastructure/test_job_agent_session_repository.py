from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.application.jobs.job_agent_sessions import JobAgentSessionRecord
from app.infrastructure.persistence.database import Base, create_session_factory
from app.infrastructure.persistence.job_agent_session_repository import (
    SqlAlchemyJobAgentSessionStore,
)


class JobAgentSessionRepositoryTest(unittest.TestCase):
    def test_reads_only_active_record_and_updates_last_seen(self) -> None:
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        store = SqlAlchemyJobAgentSessionStore(create_session_factory(engine))
        now = datetime(2026, 9, 30, tzinfo=timezone.utc)
        record = JobAgentSessionRecord(
            token_hash="a" * 64,
            owner_id="f0d4db53-214d-42fb-9f32-14a07c29adbb",
            created_at=now,
            last_seen_at=now,
            expires_at=now + timedelta(minutes=5),
        )
        try:
            store.create(record)
            self.assertEqual(record, store.find_active(token_hash=record.token_hash, now=now))
            self.assertIsNone(
                store.find_active(
                    token_hash=record.token_hash,
                    now=now + timedelta(minutes=5),
                )
            )
            seen_at = now + timedelta(minutes=1)
            store.mark_seen(token_hash=record.token_hash, seen_at=seen_at)
            self.assertEqual(
                seen_at,
                store.find_active(token_hash=record.token_hash, now=now).last_seen_at,
            )
        finally:
            engine.dispose()
