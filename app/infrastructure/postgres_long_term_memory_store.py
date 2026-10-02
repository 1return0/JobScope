"""PostgreSQL implementation of consented long-term preference memory."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import psycopg
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row

from app.application.memory.long_term_memory import (
    LongTermMemoryEntry,
    LongTermMemoryWrite,
)


class PostgresLongTermMemoryStore:
    def __init__(
        self,
        store: PostgresStore,
        *,
        application_namespace: str,
    ) -> None:
        normalized_namespace = application_namespace.strip()
        if not normalized_namespace:
            raise ValueError("memory application_namespace must not be blank")
        self._store = store
        self._namespace_prefix = (
            normalized_namespace,
            "long-term-memory",
        )

    def upsert_preference(
        self,
        write: LongTermMemoryWrite,
    ) -> LongTermMemoryEntry:
        namespace = self._namespace_for(write.owner_id)
        existing = self._store.get(namespace, write.preference_key)
        revision = (
            int(existing.value.get("revision", 0)) + 1
            if existing is not None
            else 1
        )
        self._store.put(
            namespace,
            write.preference_key,
            {
                "preference_value": write.preference_value,
                "consented_at": write.consented_at.isoformat(),
                "revision": revision,
            },
            index=False,
        )
        stored = self._store.get(namespace, write.preference_key)
        if stored is None:
            raise RuntimeError("long-term memory write was not persisted")
        return self._to_entry(write.owner_id, stored)

    def list_preferences(
        self,
        owner_id: str,
    ) -> tuple[LongTermMemoryEntry, ...]:
        normalized_owner_id = _require_owner_id(owner_id)
        stored_items = self._list_stored_items(
            self._namespace_for(normalized_owner_id)
        )
        return tuple(
            self._to_entry(normalized_owner_id, item)
            for item in sorted(stored_items, key=lambda item: item.key)
        )

    def delete_preference(
        self,
        owner_id: str,
        preference_key: str,
    ) -> bool:
        normalized_owner_id = _require_owner_id(owner_id)
        normalized_key = _require_preference_key(preference_key)
        namespace = self._namespace_for(normalized_owner_id)
        if self._store.get(namespace, normalized_key) is None:
            return False
        self._store.delete(namespace, normalized_key)
        return True

    def delete_all_preferences(self, owner_id: str) -> int:
        normalized_owner_id = _require_owner_id(owner_id)
        namespace = self._namespace_for(normalized_owner_id)
        deleted_count = 0
        while stored_items := self._store.search(namespace, limit=100):
            for item in stored_items:
                self._store.delete(namespace, item.key)
            deleted_count += len(stored_items)
        return deleted_count

    def _namespace_for(self, owner_id: str) -> tuple[str, ...]:
        return (*self._namespace_prefix, _require_owner_id(owner_id))

    def _list_stored_items(self, namespace: tuple[str, ...]) -> list[object]:
        stored_items: list[object] = []
        offset = 0
        while batch := self._store.search(namespace, limit=100, offset=offset):
            stored_items.extend(batch)
            if len(batch) < 100:
                break
            offset += len(batch)
        return stored_items

    @staticmethod
    def _to_entry(owner_id: str, stored: object) -> LongTermMemoryEntry:
        value = getattr(stored, "value")
        try:
            return LongTermMemoryEntry(
                owner_id=owner_id,
                preference_key=getattr(stored, "key"),
                preference_value=value["preference_value"],
                consented_at=datetime.fromisoformat(value["consented_at"]),
                created_at=getattr(stored, "created_at"),
                updated_at=getattr(stored, "updated_at"),
                revision=int(value["revision"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("stored long-term memory has an invalid shape") from error


@contextmanager
def open_postgres_long_term_memory_store(
    dsn: str,
    *,
    application_namespace: str,
) -> Iterator[PostgresLongTermMemoryStore]:
    if not dsn.strip():
        raise ValueError("long-term memory PostgreSQL DSN must not be blank")
    with psycopg.connect(
        dsn,
        autocommit=True,
        row_factory=dict_row,
        prepare_threshold=0,
    ) as connection:
        store = PostgresStore(connection)
        store.setup()
        yield PostgresLongTermMemoryStore(
            store,
            application_namespace=application_namespace,
        )


def _require_owner_id(owner_id: str) -> str:
    normalized_owner_id = owner_id.strip()
    if not normalized_owner_id:
        raise ValueError("memory owner_id must not be blank")
    return normalized_owner_id


def _require_preference_key(preference_key: str) -> str:
    normalized_key = preference_key.strip()
    if not normalized_key:
        raise ValueError("memory preference_key must not be blank")
    return normalized_key
