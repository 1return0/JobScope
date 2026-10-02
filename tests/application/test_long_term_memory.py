from __future__ import annotations

from datetime import datetime, timezone
from unittest import TestCase

from app.application.memory.long_term_memory import (
    LongTermMemoryEntry,
    LongTermMemoryService,
    LongTermMemoryWrite,
)


class _Store:
    def __init__(self) -> None:
        self.writes: list[LongTermMemoryWrite] = []

    def upsert_preference(
        self,
        write: LongTermMemoryWrite,
    ) -> LongTermMemoryEntry:
        self.writes.append(write)
        return LongTermMemoryEntry(
            owner_id=write.owner_id,
            preference_key=write.preference_key,
            preference_value=write.preference_value,
            consented_at=write.consented_at,
            created_at=write.consented_at,
            updated_at=write.consented_at,
            revision=1,
        )


class LongTermMemoryServiceTest(TestCase):
    def setUp(self) -> None:
        self.store = _Store()
        self.service = LongTermMemoryService(
            self.store,
            allowed_preference_keys=frozenset({"preferred_location"}),
        )

    def test_rejects_memory_write_without_explicit_consent(self) -> None:
        with self.assertRaisesRegex(ValueError, "consent"):
            self.service.remember_preference(
                owner_id="session-owner-1",
                preference_key="preferred_location",
                preference_value="上海",
                user_consented=False,
            )

        self.assertEqual([], self.store.writes)

    def test_rejects_a_key_outside_the_business_allow_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "not allowed"):
            self.service.remember_preference(
                owner_id="session-owner-1",
                preference_key="raw_ticket_content",
                preference_value="账号密码是...",
                user_consented=True,
            )

        self.assertEqual([], self.store.writes)

    def test_writes_normalized_preference_with_recorded_consent_time(self) -> None:
        consented_at = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)

        entry = self.service.remember_preference(
            owner_id=" session-owner-1 ",
            preference_key=" preferred_location ",
            preference_value=" 上海 ",
            user_consented=True,
            now=consented_at,
        )

        self.assertEqual("session-owner-1", entry.owner_id)
        self.assertEqual("preferred_location", entry.preference_key)
        self.assertEqual("上海", entry.preference_value)
        self.assertEqual(consented_at, entry.consented_at)
        self.assertEqual(1, entry.revision)
