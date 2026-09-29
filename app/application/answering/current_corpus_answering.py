from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchReport,
)
from app.application.answering.grounded_answer_service import GroundedAnswerResult
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


class CurrentCorpusAnswerService:
    def __init__(
        self,
        searcher: CurrentCorpusHybridSearcher,
        answerer: RetrievedEvidenceAnswerer,
    ) -> None:
        self._searcher = searcher
        self._answerer = answerer

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
        retrieved_chunks = tuple(
            result.chunk for result in retrieval.results
        )
        answer = self._answerer.answer(
            normalized_question,
            retrieved_chunks=retrieved_chunks,
        )
        return CurrentCorpusAnswerResult(
            retrieval=retrieval,
            answer=answer,
        )
