from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.domain.documents.document_chunking import EvidenceChunk


AnswerStatus = Literal["answered", "insufficient_evidence"]


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    evidence_id: str
    quoted_text: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("citation evidence_id must not be blank")
        if not self.quoted_text.strip():
            raise ValueError("citation quoted_text must not be blank")


@dataclass(frozen=True, slots=True)
class GroundedClaim:
    text: str
    citations: tuple[EvidenceCitation, ...]

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("claim text must not be blank")
        if not self.citations:
            raise ValueError("answered claim must contain citations")


@dataclass(frozen=True, slots=True)
class GroundedAnswerDraft:
    status: AnswerStatus
    claims: tuple[GroundedClaim, ...]
    fallback_message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("answered", "insufficient_evidence"):
            raise ValueError("unsupported answer status")
        if self.status == "answered":
            if not self.claims:
                raise ValueError("answered draft must contain claims")
            if self.fallback_message is not None:
                raise ValueError(
                    "answered draft must not contain fallback_message"
                )
            return
        if self.claims:
            raise ValueError(
                "insufficient-evidence draft must not contain claims"
            )
        if not (self.fallback_message or "").strip():
            raise ValueError(
                "insufficient-evidence draft requires fallback_message"
            )


@dataclass(frozen=True, slots=True)
class CitationValidationIssue:
    code: str
    claim_index: int
    citation_index: int
    evidence_id: str


@dataclass(frozen=True, slots=True)
class CitationValidationReport:
    valid: bool
    checked_claim_count: int
    checked_citation_count: int
    issues: tuple[CitationValidationIssue, ...]


class EvidenceGroundingValidator:
    def validate(
        self,
        draft: GroundedAnswerDraft,
        *,
        retrieved_chunks: tuple[EvidenceChunk, ...],
    ) -> CitationValidationReport:
        chunks_by_id = {
            chunk.evidence_id: chunk for chunk in retrieved_chunks
        }
        if len(chunks_by_id) != len(retrieved_chunks):
            raise ValueError(
                "retrieved evidence contains duplicate evidence IDs"
            )

        issues: list[CitationValidationIssue] = []
        checked_citation_count = 0
        for claim_index, claim in enumerate(draft.claims, start=1):
            seen_evidence_ids: set[str] = set()
            for citation_index, citation in enumerate(
                claim.citations,
                start=1,
            ):
                checked_citation_count += 1
                if citation.evidence_id in seen_evidence_ids:
                    issues.append(
                        CitationValidationIssue(
                            code="duplicate-citation-in-claim",
                            claim_index=claim_index,
                            citation_index=citation_index,
                            evidence_id=citation.evidence_id,
                        )
                    )
                    continue
                seen_evidence_ids.add(citation.evidence_id)

                chunk = chunks_by_id.get(citation.evidence_id)
                if chunk is None:
                    issues.append(
                        CitationValidationIssue(
                            code="citation-outside-retrieved-evidence",
                            claim_index=claim_index,
                            citation_index=citation_index,
                            evidence_id=citation.evidence_id,
                        )
                    )
                    continue
                if _normalize_for_quote_match(
                    citation.quoted_text
                ) not in _normalize_for_quote_match(chunk.text):
                    issues.append(
                        CitationValidationIssue(
                            code="quoted-text-not-found-in-evidence",
                            claim_index=claim_index,
                            citation_index=citation_index,
                            evidence_id=citation.evidence_id,
                        )
                    )

        return CitationValidationReport(
            valid=not issues,
            checked_claim_count=len(draft.claims),
            checked_citation_count=checked_citation_count,
            issues=tuple(issues),
        )


def _normalize_for_quote_match(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()
