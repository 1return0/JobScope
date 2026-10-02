from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchReport,
)
from app.application.answering.grounded_answer_service import GroundedAnswerResult
from app.application.jobs.controlled_replanning import (
    ControlledReplanPolicy,
    ReplanAuditRecord,
    ReplanAssessment,
)
from app.domain.documents.document_chunking import EvidenceChunk


class CurrentCorpusHybridSearcher(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_references: tuple[str, ...] | None = None,
    ) -> CurrentCorpusHybridSearchReport:
        ...


class RetrievedEvidenceAnswerer(Protocol):
    def answer(
        self,
        question: str,
        *,
        retrieved_chunks: tuple[EvidenceChunk, ...],
    ) -> GroundedAnswerResult:
        ...


@dataclass(frozen=True, slots=True)
class CurrentCorpusAnswerResult:
    retrieval: CurrentCorpusHybridSearchReport
    answer: GroundedAnswerResult
    replan: ReplanAssessment = ReplanAssessment(
        "continue",
        "not-observed",
    )
    # This is an internal audit snapshot, not a new user-controlled filter.
    replan_audit: ReplanAuditRecord | None = None


class CurrentCorpusAnswerService:
    _MAX_REPLAN_TOP_K = 10

    def __init__(
        self,
        searcher: CurrentCorpusHybridSearcher,
        answerer: RetrievedEvidenceAnswerer,
        replan_policy: ControlledReplanPolicy | None = None,
    ) -> None:
        self._searcher = searcher
        self._answerer = answerer
        self._replan_policy = replan_policy or ControlledReplanPolicy()

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        source_references: tuple[str, ...] | None = None,
    ) -> CurrentCorpusAnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        search_arguments = {"top_k": top_k}
        if source_references is not None:
            search_arguments["source_references"] = source_references
        retrieval = self._searcher.search(
            normalized_question,
            **search_arguments,
        )
        answer = self._answer_for_retrieval(
            normalized_question,
            retrieval,
        )
        replan_audit = self._replan_policy.assess_with_audit(
            original_query=normalized_question,
            original_constraints=self._source_reference_constraint(
                source_references
            ),
            result_code=answer.draft.status,
            replan_count=0,
            operation_kind="read",
        )
        replan = replan_audit.assessment
        widened_top_k = min(top_k * 2, self._MAX_REPLAN_TOP_K)
        if replan.decision == "replan_once" and widened_top_k > top_k:
            retry_arguments = {"top_k": widened_top_k}
            if source_references is not None:
                retry_arguments["source_references"] = source_references
            retrieval = self._searcher.search(
                normalized_question,
                **retry_arguments,
            )
            answer = self._answer_for_retrieval(
                normalized_question,
                retrieval,
            )
        elif replan.decision == "replan_once":
            # The policy permits a retry, but this request is already at the
            # fixed depth cap. Running the same retrieval again adds no value.
            replan = ReplanAssessment(
                "stop",
                "retrieval-depth-limit-reached",
            )
        else:
            replan = ReplanAssessment("continue", "answer-sufficient")
        return CurrentCorpusAnswerResult(
            retrieval=retrieval,
            answer=answer,
            replan=replan,
            replan_audit=replan_audit,
        )

    @staticmethod
    def _source_reference_constraint(
        source_references: tuple[str, ...] | None,
    ) -> dict[str, str]:
        """Snapshot the caller's source scope without changing it on retry."""
        if source_references is None:
            return {}
        return {"source_references": "\n".join(source_references)}

    def _answer_for_retrieval(
        self,
        question: str,
        retrieval: CurrentCorpusHybridSearchReport,
    ) -> GroundedAnswerResult:
        return self._answerer.answer(
            question,
            retrieved_chunks=tuple(result.chunk for result in retrieval.results),
        )
