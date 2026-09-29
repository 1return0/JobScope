from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchReport,
    HybridSearchCapacityError,
)


class HybridSearchProvider(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> CurrentCorpusHybridSearchReport: ...


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieverRankResponse(BaseModel):
    retriever: str
    rank: int


class HybridEvidenceResponse(BaseModel):
    rank: int
    evidence_id: str
    fusion_score: float
    ranks_by_retriever: list[RetrieverRankResponse]
    source_reference: str
    text: str
    heading_path: list[str]
    page_number: int | None


class HybridSearchResponse(BaseModel):
    query: str
    top_k: int
    candidate_k: int
    rank_constant: int
    corpus_size: int
    chunker_version: str
    tokenizer_version: str
    embedding_identity: str
    corpus_fingerprint: str
    results: list[HybridEvidenceResponse]


def build_retrieval_router(
    service: HybridSearchProvider | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])

    @router.post(
        "/hybrid",
        response_model=HybridSearchResponse,
        summary="Search the prepared current corpus with Hybrid RRF",
    )
    def search_hybrid(
        payload: HybridSearchRequest,
        request: Request,
    ) -> HybridSearchResponse:
        active_service = service or getattr(
            request.app.state,
            "hybrid_search_service",
            None,
        )
        if active_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="hybrid retrieval is disabled or not prepared",
            )
        try:
            report = active_service.search(
                payload.query,
                top_k=payload.top_k,
            )
        except HybridSearchCapacityError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": "1"},
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
        return _to_response(report)

    return router


def _to_response(
    report: CurrentCorpusHybridSearchReport,
) -> HybridSearchResponse:
    return HybridSearchResponse(
        query=report.query,
        top_k=report.top_k,
        candidate_k=report.candidate_k,
        rank_constant=report.rank_constant,
        corpus_size=report.corpus_size,
        chunker_version=report.chunker_version,
        tokenizer_version=report.tokenizer_version,
        embedding_identity=report.embedding_identity,
        corpus_fingerprint=report.corpus_fingerprint,
        results=[
            HybridEvidenceResponse(
                rank=item.rank,
                evidence_id=item.chunk.evidence_id,
                fusion_score=item.fusion_score,
                ranks_by_retriever=[
                    RetrieverRankResponse(
                        retriever=name,
                        rank=rank,
                    )
                    for name, rank in item.ranks_by_retriever
                ],
                source_reference=item.chunk.source_reference,
                text=item.chunk.text,
                heading_path=list(
                    item.chunk.location.heading_path
                ),
                page_number=item.chunk.location.page_number,
            )
            for item in report.results
        ],
    )
