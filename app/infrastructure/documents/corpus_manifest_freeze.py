from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.domain.documents.corpus_manifest import (
    VerifiedCorpusManifest,
)


FROZEN_MANIFEST_SCHEMA_VERSION = "jobscope-corpus-manifest-v1"


class FrozenManifestConflictError(ValueError):
    pass


class EmptyCorpusManifestError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FrozenManifestReport:
    manifest_identity: str
    path: Path
    created: bool


def freeze_verified_corpus_manifest(
    verified: VerifiedCorpusManifest,
    *,
    output_directory: Path,
) -> FrozenManifestReport:
    if not verified.documents:
        raise EmptyCorpusManifestError(
            "cannot freeze an empty corpus manifest"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if not output_directory.is_dir():
        raise NotADirectoryError(
            f"manifest output is not a directory: {output_directory}"
        )

    manifest_identity = verified.manifest.identity
    output_path = output_directory / f"{manifest_identity}.json"
    payload = _canonical_frozen_payload(verified)

    try:
        with output_path.open("xb") as frozen_file:
            frozen_file.write(payload)
    except FileExistsError:
        if output_path.read_bytes() != payload:
            raise FrozenManifestConflictError(
                "frozen manifest identity already exists with "
                "different content"
            )
        return FrozenManifestReport(
            manifest_identity=manifest_identity,
            path=output_path,
            created=False,
        )

    return FrozenManifestReport(
        manifest_identity=manifest_identity,
        path=output_path,
        created=True,
    )


def _canonical_frozen_payload(
    verified: VerifiedCorpusManifest,
) -> bytes:
    documents = []
    for document in sorted(
        verified.documents,
        key=lambda item: item.entry.source_id,
    ):
        documents.append(
            {
                "entry": document.entry.canonical_record(),
                "artifact": {
                    "document_format": (
                        document.artifact.document_format
                    ),
                    "byte_size": document.artifact.byte_size,
                    "content_sha256": (
                        document.artifact.content_sha256
                    ),
                },
            }
        )

    content = {
        "schema_version": FROZEN_MANIFEST_SCHEMA_VERSION,
        "manifest_identity": verified.manifest.identity,
        "document_count": len(verified.documents),
        "company_count": verified.manifest.company_count,
        "documents": documents,
    }
    return (
        json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
