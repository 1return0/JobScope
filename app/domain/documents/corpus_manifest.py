from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import urlsplit

from app.domain.documents.document_ingestion import DocumentArtifact


ManifestSourceKind: TypeAlias = Literal[
    "official_company",
    "official_university",
    "job_board",
]
ManifestStatus: TypeAlias = Literal[
    "active",
    "expired",
    "unknown",
]

_SOURCE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_SOURCE_KINDS = frozenset(
    {"official_company", "official_university", "job_board"}
)
_STATUSES = frozenset({"active", "expired", "unknown"})


def _normalize_required(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _normalize_url(value: str, *, field_name: str) -> str:
    normalized = _normalize_required(value, field_name=field_name)
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an HTTP URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain credentials")
    return normalized


@dataclass(frozen=True, slots=True)
class CorpusManifestEntry:
    source_id: str
    company: str
    job_title: str
    job_id: str | None
    recruitment_type: str
    published_date: date | None
    location: str
    source_kind: ManifestSourceKind
    source_url: str
    final_url: str
    captured_at: datetime
    artifact_path: Path
    content_sha256: str
    status: ManifestStatus
    notes: str = ""

    def __post_init__(self) -> None:
        source_id = _normalize_required(
            self.source_id,
            field_name="source_id",
        ).casefold()
        if not _SOURCE_ID_PATTERN.fullmatch(source_id):
            raise ValueError(
                "source_id must use lowercase letters, numbers, dot, "
                "underscore or hyphen"
            )
        object.__setattr__(self, "source_id", source_id)

        for field_name in (
            "company",
            "job_title",
            "recruitment_type",
            "location",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_required(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )

        if self.job_id is not None:
            normalized_job_id = self.job_id.strip() or None
            object.__setattr__(self, "job_id", normalized_job_id)

        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError("source_kind is not supported")
        if self.status not in _STATUSES:
            raise ValueError("status is not supported")

        object.__setattr__(
            self,
            "source_url",
            _normalize_url(self.source_url, field_name="source_url"),
        )
        object.__setattr__(
            self,
            "final_url",
            _normalize_url(self.final_url, field_name="final_url"),
        )

        if (
            self.captured_at.tzinfo is None
            or self.captured_at.utcoffset() is None
        ):
            raise ValueError("captured_at must include a timezone")
        if (
            self.published_date is not None
            and self.published_date > self.captured_at.date()
        ):
            raise ValueError(
                "published_date must not be after captured_at"
            )

        artifact_path = Path(self.artifact_path)
        if (
            artifact_path.is_absolute()
            or artifact_path.drive
            or ".." in artifact_path.parts
        ):
            raise ValueError(
                "artifact_path must stay relative to the artifact root"
            )
        if artifact_path in {Path(), Path(".")}:
            raise ValueError("artifact_path must identify a file")
        object.__setattr__(self, "artifact_path", artifact_path)

        normalized_sha256 = self.content_sha256.strip().casefold()
        if len(normalized_sha256) != 64:
            raise ValueError(
                "content_sha256 must be a SHA-256 hex digest"
            )
        try:
            int(normalized_sha256, 16)
        except ValueError as error:
            raise ValueError(
                "content_sha256 must be a SHA-256 hex digest"
            ) from error
        object.__setattr__(self, "content_sha256", normalized_sha256)
        object.__setattr__(self, "notes", self.notes.strip())

    def canonical_record(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "company": self.company,
            "job_title": self.job_title,
            "job_id": self.job_id or "",
            "recruitment_type": self.recruitment_type,
            "published_date": (
                self.published_date.isoformat()
                if self.published_date is not None
                else ""
            ),
            "location": self.location,
            "source_kind": self.source_kind,
            "source_url": self.source_url,
            "final_url": self.final_url,
            "captured_at": self.captured_at.isoformat(),
            "artifact_path": self.artifact_path.as_posix(),
            "content_sha256": self.content_sha256,
            "status": self.status,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    entries: tuple[CorpusManifestEntry, ...]

    def __post_init__(self) -> None:
        source_ids = [entry.source_id for entry in self.entries]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("manifest contains duplicate source_id")

    @property
    def identity(self) -> str:
        canonical_entries = [
            entry.canonical_record()
            for entry in sorted(
                self.entries,
                key=lambda item: item.source_id,
            )
        ]
        payload = json.dumps(
            canonical_entries,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def company_count(self) -> int:
        return len({entry.company.casefold() for entry in self.entries})


@dataclass(frozen=True, slots=True)
class VerifiedCorpusDocument:
    entry: CorpusManifestEntry
    artifact: DocumentArtifact

    def __post_init__(self) -> None:
        if self.entry.content_sha256 != self.artifact.content_sha256:
            raise ValueError("entry and artifact hashes must match")
        if self.entry.final_url != self.artifact.source_reference:
            raise ValueError("entry and artifact sources must match")


@dataclass(frozen=True, slots=True)
class VerifiedCorpusManifest:
    manifest: CorpusManifest
    documents: tuple[VerifiedCorpusDocument, ...]

    def __post_init__(self) -> None:
        if len(self.manifest.entries) != len(self.documents):
            raise ValueError(
                "every manifest entry must have one verified document"
            )
        if tuple(
            document.entry for document in self.documents
        ) != self.manifest.entries:
            raise ValueError(
                "verified documents must follow manifest entry order"
            )
