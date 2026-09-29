from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerResult,
)
from app.application.jobs.job_search_tool import (
    SearchCurrentJobsArguments,
    SearchCurrentJobsOutput,
    SearchCurrentJobsTool,
)


class ScopedCurrentCorpusAnswerProvider(Protocol):
    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        source_references: tuple[str, ...] | None = None,
    ) -> CurrentCorpusAnswerResult:
        ...


class RecruitmentQuestionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class RecruitmentEvidenceCitation(BaseModel):
    evidence_id: str
    quoted_text: str
    source_reference: str
    heading_path: tuple[str, ...]
    page_number: int | None


class RecruitmentEvidenceClaim(BaseModel):
    text: str
    citations: tuple[RecruitmentEvidenceCitation, ...]


class RecruitmentEvidenceTrace(BaseModel):
    top_k: int
    corpus_size: int
    corpus_fingerprint: str
    embedding_identity: str


class RecruitmentQuestionOutput(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    claims: tuple[RecruitmentEvidenceClaim, ...]
    fallback_message: str | None
    model_called: bool
    grounding_valid: bool
    retrieval: RecruitmentEvidenceTrace


class AnswerRecruitmentQuestionTool:
    name = "answer_recruitment_question"
    description = (
        "Answer questions about recruitment notices, qualifications, duties, "
        "application procedures, deadlines or other document details using "
        "hybrid RAG. Returns only evidence-grounded claims with citations. "
        "Do not use it for a request that also asks to find or filter jobs; use "
        "search_jobs_with_evidence for that mixed request."
    )

    def __init__(self, answer_service: ScopedCurrentCorpusAnswerProvider) -> None:
        self._answer_service = answer_service

    @property
    def arguments_schema(self) -> dict[str, Any]:
        return RecruitmentQuestionArguments.model_json_schema()

    def invoke(self, raw_arguments: dict[str, Any]) -> RecruitmentQuestionOutput:
        return self.execute(
            RecruitmentQuestionArguments.model_validate(raw_arguments)
        )

    def execute(
        self,
        arguments: RecruitmentQuestionArguments,
        *,
        source_references: tuple[str, ...] | None = None,
    ) -> RecruitmentQuestionOutput:
        result = self._answer_service.answer(
            arguments.question,
            top_k=arguments.top_k,
            source_references=source_references,
        )
        return _to_answer_output(result)


class SearchJobsWithEvidenceArguments(SearchCurrentJobsArguments):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description=(
            "What should be summarized from the source documents of the "
            "matched jobs. Preserve the user's requested details."
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20)


class SearchJobsWithEvidenceOutput(BaseModel):
    status: Literal["found", "no_matches"]
    jobs: SearchCurrentJobsOutput
    source_scope: tuple[str, ...]
    evidence_answer: RecruitmentQuestionOutput | None


class SearchJobsWithEvidenceTool:
    name = "search_jobs_with_evidence"
    description = (
        "Handle mixed requests that both filter/find current jobs and ask for "
        "details or a summary. First searches activated structured job records, "
        "then restricts hybrid RAG to the matched jobs' current source documents."
    )

    def __init__(
        self,
        search_tool: SearchCurrentJobsTool,
        answer_tool: AnswerRecruitmentQuestionTool,
    ) -> None:
        self._search_tool = search_tool
        self._answer_tool = answer_tool

    @property
    def arguments_schema(self) -> dict[str, Any]:
        return SearchJobsWithEvidenceArguments.model_json_schema()

    def invoke(
        self,
        raw_arguments: dict[str, Any],
    ) -> SearchJobsWithEvidenceOutput:
        return self.execute(
            SearchJobsWithEvidenceArguments.model_validate(raw_arguments)
        )

    def execute(
        self,
        arguments: SearchJobsWithEvidenceArguments,
    ) -> SearchJobsWithEvidenceOutput:
        search_arguments = SearchCurrentJobsArguments.model_validate(
            arguments.model_dump(exclude={"question", "top_k"})
        )
        jobs = self._search_tool.execute(search_arguments)
        if jobs.status == "no_matches":
            return SearchJobsWithEvidenceOutput(
                status="no_matches",
                jobs=jobs,
                source_scope=(),
                evidence_answer=None,
            )

        source_scope = tuple(
            sorted(
                {
                    job.source_reference
                    for job in jobs.jobs
                    if job.source_reference is not None
                }
            )
        )
        evidence_answer = self._answer_tool.execute(
            RecruitmentQuestionArguments(
                question=arguments.question,
                top_k=arguments.top_k,
            ),
            source_references=source_scope,
        )
        return SearchJobsWithEvidenceOutput(
            status="found",
            jobs=jobs,
            source_scope=source_scope,
            evidence_answer=evidence_answer,
        )


def _to_answer_output(
    result: CurrentCorpusAnswerResult,
) -> RecruitmentQuestionOutput:
    chunks_by_id = {
        item.chunk.evidence_id: item.chunk
        for item in result.retrieval.results
    }
    return RecruitmentQuestionOutput(
        status=result.answer.draft.status,
        claims=tuple(
            RecruitmentEvidenceClaim(
                text=claim.text,
                citations=tuple(
                    RecruitmentEvidenceCitation(
                        evidence_id=citation.evidence_id,
                        quoted_text=citation.quoted_text,
                        source_reference=(
                            chunks_by_id[citation.evidence_id].source_reference
                        ),
                        heading_path=(
                            chunks_by_id[citation.evidence_id]
                            .location.heading_path
                        ),
                        page_number=(
                            chunks_by_id[citation.evidence_id]
                            .location.page_number
                        ),
                    )
                    for citation in claim.citations
                ),
            )
            for claim in result.answer.draft.claims
        ),
        fallback_message=result.answer.draft.fallback_message,
        model_called=result.answer.model_called,
        grounding_valid=result.answer.validation.valid,
        retrieval=RecruitmentEvidenceTrace(
            top_k=result.retrieval.top_k,
            corpus_size=result.retrieval.corpus_size,
            corpus_fingerprint=result.retrieval.corpus_fingerprint,
            embedding_identity=result.retrieval.embedding_identity,
        ),
    )
