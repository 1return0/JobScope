from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from app.application.jobs.job_agent_sessions import (
    JobAgentSessionNotFoundError,
    JobAgentSessionRecord,
    JobAgentSessionService,
)


class _Store:
    def __init__(self) -> None:
        self.records: dict[str, JobAgentSessionRecord] = {}

    def create(self, record: JobAgentSessionRecord) -> None:
        self.records[record.token_hash] = record

    def find_active(self, *, token_hash: str, now: datetime):
        record = self.records.get(token_hash)
        return record if record is not None and record.expires_at > now else None

    def mark_seen(self, *, token_hash: str, seen_at: datetime) -> None:
        previous = self.records[token_hash]
        self.records[token_hash] = JobAgentSessionRecord(
            token_hash=previous.token_hash,
            owner_id=previous.owner_id,
            expires_at=previous.expires_at,
            created_at=previous.created_at,
            last_seen_at=seen_at,
        )


class JobAgentSessionServiceTest(unittest.TestCase):
    def test_issues_raw_token_but_persists_only_its_hash(self) -> None:
        store = _Store()
        service = JobAgentSessionService(store, ttl=timedelta(hours=1))
        now = datetime(2026, 9, 30, tzinfo=timezone.utc)

        issued = service.issue_session(now=now)

        self.assertNotIn(issued.session_token, store.records)
        self.assertEqual(1, len(store.records))
        record = next(iter(store.records.values()))
        self.assertNotEqual(issued.session_token, record.token_hash)
        self.assertEqual(now + timedelta(hours=1), issued.expires_at)

    def test_resolves_owner_only_from_valid_unexpired_token(self) -> None:
        store = _Store()
        service = JobAgentSessionService(store, ttl=timedelta(minutes=10))
        now = datetime(2026, 9, 30, tzinfo=timezone.utc)
        issued = service.issue_session(now=now)
        record = next(iter(store.records.values()))

        self.assertEqual(
            record.owner_id,
            service.resolve_owner_id(
                issued.session_token,
                now=now + timedelta(minutes=1),
            ),
        )
        self.assertEqual(
            now + timedelta(minutes=1),
            next(iter(store.records.values())).last_seen_at,
        )
        with self.assertRaises(JobAgentSessionNotFoundError):
            service.resolve_owner_id("invented-token", now=now)
        with self.assertRaises(JobAgentSessionNotFoundError):
            service.resolve_owner_id(
                issued.session_token,
                now=now + timedelta(minutes=10),
            )
