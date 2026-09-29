from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.domain.documents.corpus_manifest import VerifiedCorpusManifest


@dataclass(frozen=True, slots=True)
class CorpusProfile:
    manifest_identity: str
    document_count: int
    company_count: int
    total_bytes: int
    unique_content_count: int
    duplicate_content_group_count: int
    duplicate_content_document_count: int
    missing_published_date_count: int
    company_counts: dict[str, int]
    format_counts: dict[str, int]
    source_kind_counts: dict[str, int]
    status_counts: dict[str, int]


def build_corpus_profile(
    verified: VerifiedCorpusManifest,
) -> CorpusProfile:
    documents = verified.documents
    company_counts = Counter(
        document.entry.company for document in documents
    )
    format_counts = Counter(
        document.artifact.document_format for document in documents
    )
    source_kind_counts = Counter(
        document.entry.source_kind for document in documents
    )
    status_counts = Counter(
        document.entry.status for document in documents
    )
    content_counts = Counter(
        document.artifact.content_sha256 for document in documents
    )
    duplicate_groups = {
        content_sha256: count
        for content_sha256, count in content_counts.items()
        if count > 1
    }

    return CorpusProfile(
        manifest_identity=verified.manifest.identity,
        document_count=len(documents),
        company_count=len(company_counts),
        total_bytes=sum(
            document.artifact.byte_size for document in documents
        ),
        unique_content_count=len(content_counts),
        duplicate_content_group_count=len(duplicate_groups),
        duplicate_content_document_count=sum(
            duplicate_groups.values()
        ),
        missing_published_date_count=sum(
            document.entry.published_date is None
            for document in documents
        ),
        company_counts=dict(sorted(company_counts.items())),
        format_counts=dict(sorted(format_counts.items())),
        source_kind_counts=dict(sorted(source_kind_counts.items())),
        status_counts=dict(sorted(status_counts.items())),
    )
