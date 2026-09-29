from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from app.domain.documents.corpus_manifest import VerifiedCorpusManifest
from app.domain.sources.source_trust import VerifiedDomainRegistry


SourceAuditStatus = Literal[
    "verified_company_domain",
    "unverified_company_domain",
    "not_applicable",
]


@dataclass(frozen=True, slots=True)
class CorpusSourceAuditItem:
    source_id: str
    company: str
    source_kind: str
    candidate_host: str
    audit_status: SourceAuditStatus
    matched_domain: str | None = None
    evidence_reference: str | None = None
    verified_at: str | None = None
    verified_by: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusSourceAuditReport:
    manifest_identity: str
    total: int
    official_company_claims: int
    verified_company_sources: int
    unverified_company_sources: int
    not_applicable: int
    items: tuple[CorpusSourceAuditItem, ...]


def audit_corpus_sources(
    verified: VerifiedCorpusManifest,
    registry: VerifiedDomainRegistry,
) -> CorpusSourceAuditReport:
    items: list[CorpusSourceAuditItem] = []
    for document in verified.documents:
        entry = document.entry
        candidate_host = urlsplit(entry.final_url).hostname
        assert candidate_host is not None

        if entry.source_kind != "official_company":
            items.append(
                CorpusSourceAuditItem(
                    source_id=entry.source_id,
                    company=entry.company,
                    source_kind=entry.source_kind,
                    candidate_host=candidate_host,
                    audit_status="not_applicable",
                )
            )
            continue

        record = registry.find_matching_record(
            entry.company,
            candidate_host,
        )
        if record is None:
            items.append(
                CorpusSourceAuditItem(
                    source_id=entry.source_id,
                    company=entry.company,
                    source_kind=entry.source_kind,
                    candidate_host=candidate_host,
                    audit_status="unverified_company_domain",
                )
            )
            continue

        items.append(
            CorpusSourceAuditItem(
                source_id=entry.source_id,
                company=entry.company,
                source_kind=entry.source_kind,
                candidate_host=candidate_host,
                audit_status="verified_company_domain",
                matched_domain=record.domain,
                evidence_reference=record.evidence_reference,
                verified_at=record.verified_at.isoformat(),
                verified_by=record.verified_by,
            )
        )

    return CorpusSourceAuditReport(
        manifest_identity=verified.manifest.identity,
        total=len(items),
        official_company_claims=sum(
            item.source_kind == "official_company" for item in items
        ),
        verified_company_sources=sum(
            item.audit_status == "verified_company_domain"
            for item in items
        ),
        unverified_company_sources=sum(
            item.audit_status == "unverified_company_domain"
            for item in items
        ),
        not_applicable=sum(
            item.audit_status == "not_applicable" for item in items
        ),
        items=tuple(items),
    )
