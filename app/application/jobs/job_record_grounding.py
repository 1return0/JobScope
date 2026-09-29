from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    StructuredJobRecordDraft,
)


@dataclass(frozen=True, slots=True)
class JobRecordCitationIssue:
    code: str
    field_path: str
    citation_index: int
    evidence_id: str


@dataclass(frozen=True, slots=True)
class JobRecordGroundingReport:
    valid: bool
    checked_fact_count: int
    checked_citation_count: int
    issues: tuple[JobRecordCitationIssue, ...]


class JobRecordGroundingValidator:
    def validate(
        self,
        record: StructuredJobRecordDraft,
        *,
        allowed_chunks: tuple[EvidenceChunk, ...],
    ) -> JobRecordGroundingReport:
        chunks_by_id = {
            chunk.evidence_id: chunk for chunk in allowed_chunks
        }
        if len(chunks_by_id) != len(allowed_chunks):
            raise ValueError(
                "job extraction evidence contains duplicate evidence IDs"
            )

        facts = tuple(_iter_facts(record))
        issues: list[JobRecordCitationIssue] = []
        checked_citation_count = 0
        for field_path, fact in facts:
            for citation_index, citation in enumerate(
                fact.citations,
                start=1,
            ):
                checked_citation_count += 1
                chunk = chunks_by_id.get(citation.evidence_id)
                if chunk is None:
                    issues.append(
                        JobRecordCitationIssue(
                            code="citation-outside-extraction-evidence",
                            field_path=field_path,
                            citation_index=citation_index,
                            evidence_id=citation.evidence_id,
                        )
                    )
                    continue
                if _normalize_quote(citation.quoted_text) not in (
                    _normalize_quote(chunk.text)
                ):
                    issues.append(
                        JobRecordCitationIssue(
                            code="quoted-text-not-found-in-evidence",
                            field_path=field_path,
                            citation_index=citation_index,
                            evidence_id=citation.evidence_id,
                        )
                    )

        return JobRecordGroundingReport(
            valid=not issues,
            checked_fact_count=len(facts),
            checked_citation_count=checked_citation_count,
            issues=tuple(issues),
        )


def _iter_facts(record: StructuredJobRecordDraft):
    for field_name in (
        "company",
        "job_title",
        "education_requirement",
        "major_requirement",
        "recruitment_type",
        "application_deadline",
    ):
        yield field_name, getattr(record, field_name)

    for field_name in (
        "locations",
        "responsibilities",
        "required_qualifications",
        "preferred_qualifications",
    ):
        for index, fact in enumerate(getattr(record, field_name)):
            yield f"{field_name}[{index}]", fact


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()
