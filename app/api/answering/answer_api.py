from __future__ import annotations

from typing import Literal, Protocol

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.application.answering.answer_output_decoding import (
    AnswerOutputDecodingError,
)
from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerResult,
)
from app.application.retrieval.current_corpus_hybrid_search import (
    HybridSearchCapacityError,
)
from app.application.answering.grounded_answer_service import (
    UngroundedModelAnswerError,
)
from app.infrastructure.llm.openai_compatible_answer_model import (
    AnswerModelAuthenticationError,
    AnswerModelRateLimitError,
    AnswerModelResponseError,
    AnswerModelUnavailableError,
    AnswerModelUpstreamError,
)


class CurrentCorpusAnswerProvider(Protocol):
    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> CurrentCorpusAnswerResult:
        ...


class CurrentCorpusAnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class AnswerCitationResponse(BaseModel):
    evidence_id: str
    quoted_text: str
    source_reference: str
    heading_path: list[str]
    page_number: int | None


class AnswerClaimResponse(BaseModel):
    text: str
    citations: list[AnswerCitationResponse]


class AnswerRetrievalTraceResponse(BaseModel):
    top_k: int
    corpus_size: int
    corpus_fingerprint: str
    embedding_identity: str


class CurrentCorpusAnswerResponse(BaseModel):
    question: str
    status: Literal["answered", "insufficient_evidence"]
    claims: list[AnswerClaimResponse]
    fallback_message: str | None
    model_called: bool
    grounding_valid: bool
    retrieval: AnswerRetrievalTraceResponse


def build_answer_router(
    service: CurrentCorpusAnswerProvider | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1/answers", tags=["answers"])

    @router.post(
        "/grounded",
        response_model=CurrentCorpusAnswerResponse,
        summary="Answer from the prepared current corpus with citations",
    )
    def answer_from_current_corpus(
        payload: CurrentCorpusAnswerRequest,
        request: Request,
    ) -> CurrentCorpusAnswerResponse:
        active_service = service or getattr(
            request.app.state,
            "current_corpus_answer_service",
            None,
        )
        if active_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="grounded answer service is disabled or not prepared",
            )
        try:
            result = active_service.answer(
                payload.question,
                top_k=payload.top_k,
            )
        except HybridSearchCapacityError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": "1"},
            ) from error
        except AnswerModelRateLimitError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="answer model is temporarily rate limited",
                headers={"Retry-After": "1"},
            ) from error
        except (
            AnswerModelAuthenticationError,
            AnswerModelUnavailableError,
        ) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="answer model is temporarily unavailable",
            ) from error
        except (
            AnswerOutputDecodingError,
            AnswerModelResponseError,
            UngroundedModelAnswerError,
        ) as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="answer model returned an invalid grounded response",
            ) from error
        except AnswerModelUpstreamError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="answer model upstream request failed",
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        return _to_response(result)

    return router


def _to_response(
    result: CurrentCorpusAnswerResult,
) -> CurrentCorpusAnswerResponse:
    chunks_by_id = {
        item.chunk.evidence_id: item.chunk
        for item in result.retrieval.results
    }
    return CurrentCorpusAnswerResponse(
        question=result.retrieval.query,
        status=result.answer.draft.status,
        claims=[
            AnswerClaimResponse(
                text=claim.text,
                citations=[
                    AnswerCitationResponse(
                        evidence_id=citation.evidence_id,
                        quoted_text=citation.quoted_text,
                        source_reference=(
                            chunks_by_id[citation.evidence_id]
                            .source_reference
                        ),
                        heading_path=list(
                            chunks_by_id[citation.evidence_id]
                            .location.heading_path
                        ),
                        page_number=(
                            chunks_by_id[citation.evidence_id]
                            .location.page_number
                        ),
                    )
                    for citation in claim.citations
                ],
            )
            for claim in result.answer.draft.claims
        ],
        fallback_message=result.answer.draft.fallback_message,
        model_called=result.answer.model_called,
        grounding_valid=result.answer.validation.valid,
        retrieval=AnswerRetrievalTraceResponse(
            top_k=result.retrieval.top_k,
            corpus_size=result.retrieval.corpus_size,
            corpus_fingerprint=result.retrieval.corpus_fingerprint,
            embedding_identity=result.retrieval.embedding_identity,
        ),
    )
