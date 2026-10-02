"""A bounded, checkpoint-safe transcript contract for Agent conversations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


ConversationRole = Literal["user", "assistant"]
_MAX_TURNS = 12  # Six complete user/assistant rounds.
_MAX_CONTENT_CHARS = 600
_BEARER_TOKEN = re.compile(r"(?i)bearer\s+[a-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class SafeConversationTurn:
    role: ConversationRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise ValueError("conversation role is invalid")
        if not self.content.strip():
            raise ValueError("conversation content must not be blank")
        if len(self.content) > _MAX_CONTENT_CHARS:
            raise ValueError("conversation content exceeds the safe limit")

    def to_checkpoint(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def safe_conversation_turn(
    *,
    role: ConversationRole,
    content: str,
) -> SafeConversationTurn:
    normalized = _BEARER_TOKEN.sub("[redacted bearer token]", content).strip()
    if len(normalized) > _MAX_CONTENT_CHARS:
        normalized = normalized[:_MAX_CONTENT_CHARS].rstrip() + "…"
    return SafeConversationTurn(role=role, content=normalized)


def merge_safe_conversation_turns(
    existing: list[dict[str, str] | str],
    incoming: list[dict[str, str] | str],
) -> list[dict[str, str]]:
    normalized = [_normalize_checkpoint_turn(item) for item in (*existing, *incoming)]
    return normalized[-_MAX_TURNS:]


def _normalize_checkpoint_turn(item: dict[str, str] | str) -> dict[str, str]:
    # Old checkpoints stored plain user strings; migrate them safely on read.
    if isinstance(item, str):
        return safe_conversation_turn(role="user", content=item).to_checkpoint()
    return SafeConversationTurn(role=item["role"], content=item["content"]).to_checkpoint()


def safe_agent_summary(
    *,
    outcome: str,
    reason: str,
) -> SafeConversationTurn:
    """Stores a bounded summary, never raw tool output or internal trace."""
    return safe_conversation_turn(
        role="assistant",
        content=f"{outcome}: {reason}",
    )
