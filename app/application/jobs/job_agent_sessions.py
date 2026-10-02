from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Protocol
from uuid import uuid4


class JobAgentSessionNotFoundError(ValueError):
    """The browser did not present a valid, unexpired server session."""


@dataclass(frozen=True, slots=True)
class JobAgentSessionRecord:
    token_hash: str
    owner_id: str
    expires_at: datetime
    created_at: datetime
    last_seen_at: datetime


@dataclass(frozen=True, slots=True)
class IssuedJobAgentSession:
    """Raw token exists only between issuance and the HTTP-only cookie write."""

    session_token: str
    expires_at: datetime


class JobAgentSessionStore(Protocol):
    def create(self, record: JobAgentSessionRecord) -> None:
        ...

    def find_active(
        self,
        *,
        token_hash: str,
        now: datetime,
    ) -> JobAgentSessionRecord | None:
        ...

    def mark_seen(self, *, token_hash: str, seen_at: datetime) -> None:
        ...


class JobAgentSessionService:
    """Issues opaque browser sessions and resolves trusted internal owners."""

    def __init__(
        self,
        store: JobAgentSessionStore,
        *,
        ttl: timedelta,
    ) -> None:
        if ttl.total_seconds() <= 0:
            raise ValueError("job agent session ttl must be positive")
        self._store = store
        self._ttl = ttl

    def issue_session(
        self,
        *,
        now: datetime | None = None,
    ) -> IssuedJobAgentSession:
        issued_at = _utc_now() if now is None else _require_utc(now)
        token = secrets.token_urlsafe(32)
        expires_at = issued_at + self._ttl
        self._store.create(
            JobAgentSessionRecord(
                token_hash=_token_hash(token),
                owner_id=str(uuid4()),
                expires_at=expires_at,
                created_at=issued_at,
                last_seen_at=issued_at,
            )
        )
        return IssuedJobAgentSession(
            session_token=token,
            expires_at=expires_at,
        )

    def resolve_owner_id(
        self,
        session_token: str | None,
        *,
        now: datetime | None = None,
    ) -> str:
        if not session_token or not session_token.strip():
            raise JobAgentSessionNotFoundError("job agent session is required")
        resolved_at = _utc_now() if now is None else _require_utc(now)
        token_hash = _token_hash(session_token)
        record = self._store.find_active(
            token_hash=token_hash,
            now=resolved_at,
        )
        if record is None:
            raise JobAgentSessionNotFoundError(
                "job agent session is invalid or expired"
            )
        self._store.mark_seen(token_hash=token_hash, seen_at=resolved_at)
        return record.owner_id


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("session timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)
