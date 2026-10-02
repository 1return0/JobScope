from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from uuid import uuid4

from app.application.memory.long_term_memory import LongTermMemoryWrite
from app.config import load_settings
from app.infrastructure.persistence.database import build_postgresql_dsn
from app.infrastructure.postgres_long_term_memory_store import (
    open_postgres_long_term_memory_store,
)


@unittest.skipUnless(
    os.getenv("JOBSCOPE_RUN_POSTGRES_MEMORY_SMOKE") == "true",
    "set JOBSCOPE_RUN_POSTGRES_MEMORY_SMOKE=true to run PostgreSQL smoke",
)
class PostgresLongTermMemoryStoreSmokeTest(unittest.TestCase):
    def test_upsert_list_and_delete_stay_within_one_owner_namespace(self) -> None:
        owner_id = f"jobscope-memory-smoke-{uuid4()}"
        first_consent = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
        changed_consent = datetime(2026, 9, 30, 10, 0, tzinfo=timezone.utc)

        with open_postgres_long_term_memory_store(
            build_postgresql_dsn(load_settings()),
            application_namespace="jobscope",
        ) as store:
            try:
                first = store.upsert_preference(
                    LongTermMemoryWrite(
                        owner_id=owner_id,
                        preference_key="preferred_location",
                        preference_value="上海",
                        consented_at=first_consent,
                    )
                )
                changed = store.upsert_preference(
                    LongTermMemoryWrite(
                        owner_id=owner_id,
                        preference_key="preferred_location",
                        preference_value="北京",
                        consented_at=changed_consent,
                    )
                )
                stored = store.list_preferences(owner_id)

                self.assertEqual(1, first.revision)
                self.assertEqual(2, changed.revision)
                self.assertEqual(first.created_at, changed.created_at)
                self.assertEqual("北京", stored[0].preference_value)
                self.assertEqual(changed_consent, stored[0].consented_at)
                self.assertTrue(
                    store.delete_preference(owner_id, "preferred_location")
                )
                self.assertFalse(
                    store.delete_preference(owner_id, "preferred_location")
                )
            finally:
                store.delete_all_preferences(owner_id)

    def test_owner_namespaces_are_isolated_and_bulk_delete_is_scoped(self) -> None:
        first_owner_id = f"jobscope-memory-smoke-{uuid4()}"
        second_owner_id = f"jobscope-memory-smoke-{uuid4()}"
        consented_at = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)

        with open_postgres_long_term_memory_store(
            build_postgresql_dsn(load_settings()),
            application_namespace="jobscope",
        ) as store:
            try:
                store.upsert_preference(
                    LongTermMemoryWrite(
                        owner_id=first_owner_id,
                        preference_key="preferred_location",
                        preference_value="上海",
                        consented_at=consented_at,
                    )
                )
                store.upsert_preference(
                    LongTermMemoryWrite(
                        owner_id=first_owner_id,
                        preference_key="preferred_recruitment_type",
                        preference_value="校园招聘",
                        consented_at=consented_at,
                    )
                )
                store.upsert_preference(
                    LongTermMemoryWrite(
                        owner_id=second_owner_id,
                        preference_key="preferred_location",
                        preference_value="北京",
                        consented_at=consented_at,
                    )
                )

                self.assertEqual(2, store.delete_all_preferences(first_owner_id))
                self.assertEqual((), store.list_preferences(first_owner_id))
                self.assertEqual(
                    ("北京",),
                    tuple(
                        entry.preference_value
                        for entry in store.list_preferences(second_owner_id)
                    ),
                )
            finally:
                store.delete_all_preferences(first_owner_id)
                store.delete_all_preferences(second_owner_id)


if __name__ == "__main__":
    unittest.main()
