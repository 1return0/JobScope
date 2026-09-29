from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from app.domain.documents.corpus_manifest import (
    CorpusManifest,
    CorpusManifestEntry,
    ManifestSourceKind,
    ManifestStatus,
    VerifiedCorpusDocument,
    VerifiedCorpusManifest,
)
from app.domain.documents.document_ingestion import (
    DocumentArtifact,
    DocumentFormat,
    inspect_document_artifact,
)


MANIFEST_COLUMNS = (
    "source_id",
    "company",
    "job_title",
    "job_id",
    "recruitment_type",
    "published_date",
    "location",
    "source_kind",
    "source_url",
    "final_url",
    "captured_at",
    "artifact_path",
    "content_sha256",
    "status",
    "notes",
)


class CorpusManifestCsvError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CorpusArtifactRegistrationReport:
    entry: CorpusManifestEntry
    document_format: DocumentFormat
    byte_size: int
    manifest_identity: str
    document_count: int
    operation: Literal["created", "updated"]
    previous_content_sha256: str | None

    @property
    def content_changed(self) -> bool:
        return (
            self.previous_content_sha256 is not None
            and self.previous_content_sha256
            != self.entry.content_sha256
        )


def load_verified_corpus_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
) -> VerifiedCorpusManifest:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"corpus manifest does not exist: {manifest_path}"
        )
    if not artifact_root.is_dir():
        raise NotADirectoryError(
            f"artifact root does not exist: {artifact_root}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != MANIFEST_COLUMNS:
            raise CorpusManifestCsvError(
                "manifest columns must exactly match the documented schema"
            )

        entries: list[CorpusManifestEntry] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                entries.append(_parse_entry(row))
            except (TypeError, ValueError) as error:
                raise CorpusManifestCsvError(
                    f"invalid corpus manifest row {row_number}: {error}"
                ) from error

    try:
        manifest = CorpusManifest(tuple(entries))
    except ValueError as error:
        raise CorpusManifestCsvError(str(error)) from error

    resolved_root = artifact_root.resolve()
    documents = tuple(
        _verify_document(entry, resolved_root=resolved_root)
        for entry in manifest.entries
    )
    return VerifiedCorpusManifest(
        manifest=manifest,
        documents=documents,
    )


def register_corpus_artifact(
    manifest_path: Path,
    *,
    artifact_root: Path,
    artifact_path: Path,
    source_id: str,
    company: str,
    job_title: str,
    job_id: str | None,
    recruitment_type: str,
    published_date: date | None,
    location: str,
    source_kind: ManifestSourceKind,
    source_url: str,
    final_url: str,
    captured_at: datetime,
    status: ManifestStatus,
    notes: str = "",
    replace_existing: bool = False,
) -> CorpusArtifactRegistrationReport:
    verified = load_verified_corpus_manifest(
        manifest_path,
        artifact_root=artifact_root,
    )
    resolved_root = artifact_root.resolve()
    resolved_path = _resolve_artifact_path(
        artifact_path,
        resolved_root=resolved_root,
        source_id=source_id,
    )
    if not resolved_path.is_file():
        raise CorpusManifestCsvError(
            f"artifact does not exist for source_id: {source_id}"
        )

    artifact = inspect_document_artifact(
        resolved_path,
        source_reference=final_url,
    )
    entry = CorpusManifestEntry(
        source_id=source_id,
        company=company,
        job_title=job_title,
        job_id=job_id,
        recruitment_type=recruitment_type,
        published_date=published_date,
        location=location,
        source_kind=source_kind,
        source_url=source_url,
        final_url=final_url,
        captured_at=captured_at,
        artifact_path=artifact_path,
        content_sha256=artifact.content_sha256,
        status=status,
        notes=notes,
    )
    new_document = VerifiedCorpusDocument(
        entry=entry,
        artifact=artifact,
    )
    updated_verified, previous_entry = _merge_verified_document(
        verified,
        new_document,
        replace_existing=replace_existing,
    )
    _replace_manifest_csv(
        manifest_path,
        updated_verified.manifest,
    )
    return CorpusArtifactRegistrationReport(
        entry=entry,
        document_format=artifact.document_format,
        byte_size=artifact.byte_size,
        manifest_identity=updated_verified.manifest.identity,
        document_count=len(updated_verified.documents),
        operation=(
            "updated" if previous_entry is not None else "created"
        ),
        previous_content_sha256=(
            previous_entry.content_sha256
            if previous_entry is not None
            else None
        ),
    )


def _merge_verified_document(
    verified: VerifiedCorpusManifest,
    new_document: VerifiedCorpusDocument,
    *,
    replace_existing: bool,
) -> tuple[VerifiedCorpusManifest, CorpusManifestEntry | None]:
    new_entry = new_document.entry
    matching_index = next(
        (
            index
            for index, document in enumerate(verified.documents)
            if document.entry.source_id == new_entry.source_id
        ),
        None,
    )

    if matching_index is None:
        if replace_existing:
            raise CorpusManifestCsvError(
                "cannot replace a source_id that is not registered"
            )
        entries = (*verified.manifest.entries, new_entry)
        documents = (*verified.documents, new_document)
        previous_entry = None
    else:
        if not replace_existing:
            raise CorpusManifestCsvError(
                "manifest contains duplicate source_id"
            )
        previous_entry = verified.documents[matching_index].entry
        _validate_source_update(previous_entry, new_entry)
        entries_list = list(verified.manifest.entries)
        documents_list = list(verified.documents)
        entries_list[matching_index] = new_entry
        documents_list[matching_index] = new_document
        entries = tuple(entries_list)
        documents = tuple(documents_list)

    updated_manifest = CorpusManifest(entries)
    return (
        VerifiedCorpusManifest(
            manifest=updated_manifest,
            documents=documents,
        ),
        previous_entry,
    )


def _validate_source_update(
    previous: CorpusManifestEntry,
    updated: CorpusManifestEntry,
) -> None:
    if updated.captured_at <= previous.captured_at:
        raise CorpusManifestCsvError(
            "updated source must have a later captured_at"
        )
    if updated.company.casefold() != previous.company.casefold():
        raise CorpusManifestCsvError(
            "updated source must keep the same company"
        )
    if (
        updated.source_url != previous.source_url
        or updated.final_url != previous.final_url
    ):
        raise CorpusManifestCsvError(
            "updated source must keep the same source URLs"
        )
    if updated.source_kind != previous.source_kind:
        raise CorpusManifestCsvError(
            "updated source must keep the same source kind"
        )


def _parse_entry(row: dict[str, str | None]) -> CorpusManifestEntry:
    def value(column: str) -> str:
        raw_value = row.get(column)
        if raw_value is None:
            raise ValueError(f"{column} is missing")
        return raw_value.strip()

    published_text = value("published_date")
    captured_text = value("captured_at")
    if captured_text.endswith("Z"):
        captured_text = f"{captured_text[:-1]}+00:00"

    return CorpusManifestEntry(
        source_id=value("source_id"),
        company=value("company"),
        job_title=value("job_title"),
        job_id=value("job_id") or None,
        recruitment_type=value("recruitment_type"),
        published_date=(
            date.fromisoformat(published_text)
            if published_text
            else None
        ),
        location=value("location"),
        source_kind=value("source_kind"),  # type: ignore[arg-type]
        source_url=value("source_url"),
        final_url=value("final_url"),
        captured_at=datetime.fromisoformat(captured_text),
        artifact_path=Path(value("artifact_path")),
        content_sha256=value("content_sha256"),
        status=value("status"),  # type: ignore[arg-type]
        notes=value("notes"),
    )


def _verify_document(
    entry: CorpusManifestEntry,
    *,
    resolved_root: Path,
) -> VerifiedCorpusDocument:
    resolved_path = _resolve_artifact_path(
        entry.artifact_path,
        resolved_root=resolved_root,
        source_id=entry.source_id,
    )
    if not resolved_path.is_file():
        raise CorpusManifestCsvError(
            f"artifact does not exist for source_id: {entry.source_id}"
        )

    artifact = inspect_document_artifact(
        resolved_path,
        source_reference=entry.final_url,
    )
    if artifact.content_sha256 != entry.content_sha256:
        raise CorpusManifestCsvError(
            f"artifact hash mismatch for source_id: {entry.source_id}"
        )
    return VerifiedCorpusDocument(entry=entry, artifact=artifact)


def _resolve_artifact_path(
    artifact_path: Path,
    *,
    resolved_root: Path,
    source_id: str,
) -> Path:
    resolved_path = (resolved_root / artifact_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise CorpusManifestCsvError(
            f"artifact escapes configured root: {source_id}"
        ) from error
    return resolved_path


def _replace_manifest_csv(
    manifest_path: Path,
    manifest: CorpusManifest,
) -> None:
    temporary_path = manifest_path.with_name(
        f".{manifest_path.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=MANIFEST_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(
                entry.canonical_record()
                for entry in manifest.entries
            )
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
