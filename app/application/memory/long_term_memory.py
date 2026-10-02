"""Contracts for explicitly consented, stable user preferences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LongTermMemoryWrite:
    """A requested preference write after the user has explicitly consented."""

    owner_id: str
    preference_key: str
    preference_value: str
    consented_at: datetime

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("memory owner_id must not be blank")
        if not self.preference_key.strip():
            raise ValueError("memory preference_key must not be blank")
        if not self.preference_value.strip():
            raise ValueError("memory preference_value must not be blank")
        if self.consented_at.tzinfo is None:
            raise ValueError("memory consented_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class LongTermMemoryEntry:
    """A durable preference record returned by a long-term memory Store."""

    owner_id: str
    preference_key: str
    preference_value: str
    consented_at: datetime
    created_at: datetime
    updated_at: datetime
    revision: int

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("memory owner_id must not be blank")
        if not self.preference_key.strip():
            raise ValueError("memory preference_key must not be blank")
        if not self.preference_value.strip():
            raise ValueError("memory preference_value must not be blank")
        if any(
            timestamp.tzinfo is None
            for timestamp in (
                self.consented_at,
                self.created_at,
                self.updated_at,
            )
        ):
            raise ValueError("memory timestamps must be timezone-aware")
        if self.revision < 1:
            raise ValueError("memory revision must be positive")


class LongTermMemoryStore(Protocol):
    def upsert_preference(
        self,
        write: LongTermMemoryWrite,
    ) -> LongTermMemoryEntry:
        ...

    def list_preferences(
        self,
        owner_id: str,
    ) -> tuple[LongTermMemoryEntry, ...]:
        ...

    def delete_preference(
        self,
        owner_id: str,
        preference_key: str,
    ) -> bool:
        ...

    def delete_all_preferences(self, owner_id: str) -> int:
        ...


class LongTermMemoryService:
    """Applies consent and allow-list rules before a Store can write memory."""

    def __init__(
        self,
        store: LongTermMemoryStore,
        *,
        allowed_preference_keys: frozenset[str],
    ) -> None:
        if not allowed_preference_keys:
            raise ValueError("at least one memory preference key is required")
        self._store = store
        self._allowed_preference_keys = frozenset(
            key.strip() for key in allowed_preference_keys if key.strip()
        )
        if not self._allowed_preference_keys:
            raise ValueError("memory preference keys must not be blank")

    def remember_preference(
        self,
        *,
        owner_id: str,
        preference_key: str,
        preference_value: str,
        user_consented: bool,
        now: datetime | None = None,
    ) -> LongTermMemoryEntry:
        if not user_consented:
            raise ValueError("explicit user consent is required for long-term memory")
        normalized_key = preference_key.strip()
        if normalized_key not in self._allowed_preference_keys:
            raise ValueError("memory preference_key is not allowed")
        consented_at = now or datetime.now(timezone.utc)
        return self._store.upsert_preference(
            LongTermMemoryWrite(
                owner_id=owner_id.strip(),
                preference_key=normalized_key,
                preference_value=preference_value.strip(),
                consented_at=consented_at,
            )
        )

    def list_preferences(
        self,
        *,
        owner_id: str,
    ) -> tuple[LongTermMemoryEntry, ...]:
        return self._store.list_preferences(owner_id.strip())

    def forget_preference(
        self,
        *,
        owner_id: str,
        preference_key: str,
    ) -> bool:
        normalized_key = preference_key.strip()
        if normalized_key not in self._allowed_preference_keys:
            raise ValueError("memory preference_key is not allowed")
        return self._store.delete_preference(
            owner_id.strip(),
            normalized_key,
        )

    def forget_all_preferences(self, *, owner_id: str) -> int:
        return self._store.delete_all_preferences(owner_id.strip())
