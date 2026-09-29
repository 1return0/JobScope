from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from app.domain.documents.document_ingestion import DocumentFormat


@dataclass(frozen=True, slots=True)
class DocumentSnapshot:
    snapshot_id: str
    source_reference: str
    document_format: DocumentFormat
    content_sha256: str
    byte_size: int
    version_number: int
    first_observed_at: datetime
    last_observed_at: datetime
    is_current: bool

    def __post_init__(self) -> None:
        if not self.snapshot_id.startswith("snap_"):
            raise ValueError("snapshot_id must start with snap_")
        if len(self.snapshot_id) != 69:
            raise ValueError(
                "snapshot_id must contain a full SHA-256 digest"
            )
        try:
            int(self.snapshot_id.removeprefix("snap_"), 16)
            int(self.content_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "snapshot identifiers must contain SHA-256 hex digests"
            ) from error
        if len(self.content_sha256) != 64:
            raise ValueError(
                "content_sha256 must contain a full SHA-256 digest"
            )
        if not self.source_reference.strip():
            raise ValueError("source_reference must not be blank")
        if self.byte_size < 0:
            raise ValueError("byte_size must not be negative")
        if self.version_number < 1:
            raise ValueError("version_number must be positive")
        if (
            self.first_observed_at.tzinfo is None
            or self.last_observed_at.tzinfo is None
        ):
            raise ValueError("observation times must be timezone-aware")
        if self.last_observed_at < self.first_observed_at:
            raise ValueError(
                "last_observed_at must not precede first_observed_at"
            )


def build_snapshot_id(
    *,
    source_reference: str,
    content_sha256: str,
) -> str:
    identity_payload = {
        "content_sha256": content_sha256,
        "source_reference": source_reference.strip(),
    }
    canonical_json = json.dumps(
        identity_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return f"snap_{digest}"
