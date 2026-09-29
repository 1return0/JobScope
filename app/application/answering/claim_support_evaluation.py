from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.application.answering.grounded_answering import GroundedAnswerDraft
from app.domain.documents.document_chunking import EvidenceChunk


ClaimSupportLabel = Literal[
    "supported",
    "unsupported",
    "insufficient_context",
]


@dataclass(frozen=True, slots=True)
class ClaimSupportEvaluationInput:
    claim_index: int
    claim_text: str
    evidence_texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClaimSupportJudgment:
    label: ClaimSupportLabel
    rationale: str

    def __post_init__(self) -> None:
        if self.label not in (
            "supported",
            "unsupported",
            "insufficient_context",
        ):
            raise ValueError("unsupported claim-support label")
        if not self.rationale.strip():
            raise ValueError("claim-support rationale must not be blank")


class ClaimSupportJudge(Protocol):
    def judge(
        self,
        evaluation_input: ClaimSupportEvaluationInput,
    ) -> ClaimSupportJudgment:
        ...


@dataclass(frozen=True, slots=True)
class ClaimSupportCaseResult:
    claim_index: int
    claim_text: str
    judgment: ClaimSupportJudgment


@dataclass(frozen=True, slots=True)
class ClaimSupportEvaluationReport:
    claim_count: int
    supported_claim_count: int
    support_rate: float
    review_required: bool
    claims: tuple[ClaimSupportCaseResult, ...]


class GroundedClaimSupportEvaluator:
    def __init__(self, judge: ClaimSupportJudge) -> None:
        self._judge = judge

    def evaluate(
        self,
        draft: GroundedAnswerDraft,
        *,
        retrieved_chunks: tuple[EvidenceChunk, ...],
    ) -> ClaimSupportEvaluationReport:
        if draft.status != "answered":
            raise ValueError(
                "claim support evaluation requires an answered draft"
            )
        chunks_by_id = {
            chunk.evidence_id: chunk for chunk in retrieved_chunks
        }
        if len(chunks_by_id) != len(retrieved_chunks):
            raise ValueError(
                "retrieved evidence contains duplicate evidence IDs"
            )

        results = tuple(
            self._evaluate_claim(
                claim_index,
                claim.text,
                tuple(
                    chunks_by_id[citation.evidence_id].text
                    for citation in claim.citations
                    if citation.evidence_id in chunks_by_id
                ),
            )
            for claim_index, claim in enumerate(draft.claims, start=1)
        )
        supported_count = sum(
            result.judgment.label == "supported" for result in results
        )
        return ClaimSupportEvaluationReport(
            claim_count=len(results),
            supported_claim_count=supported_count,
            support_rate=supported_count / len(results),
            review_required=any(
                result.judgment.label != "supported"
                for result in results
            ),
            claims=results,
        )

    def _evaluate_claim(
        self,
        claim_index: int,
        claim_text: str,
        evidence_texts: tuple[str, ...],
    ) -> ClaimSupportCaseResult:
        if not evidence_texts:
            judgment = ClaimSupportJudgment(
                label="insufficient_context",
                rationale="no cited evidence is available for judging",
            )
        else:
            judgment = self._judge.judge(
                ClaimSupportEvaluationInput(
                    claim_index=claim_index,
                    claim_text=claim_text,
                    evidence_texts=evidence_texts,
                )
            )
        return ClaimSupportCaseResult(
            claim_index=claim_index,
            claim_text=claim_text,
            judgment=judgment,
        )
