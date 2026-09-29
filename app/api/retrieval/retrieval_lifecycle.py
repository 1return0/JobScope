from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchService,
)
from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerService,
)


HybridSearchServiceFactory = Callable[
    [],
    CurrentCorpusHybridSearchService,
]
AnswerServiceFactory = Callable[
    [CurrentCorpusHybridSearchService],
    CurrentCorpusAnswerService,
]


def build_retrieval_lifespan(
    *,
    enabled: bool,
    service_factory: HybridSearchServiceFactory,
    answer_enabled: bool = False,
    answer_service_factory: AnswerServiceFactory | None = None,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            if answer_enabled and not enabled:
                raise ValueError(
                    "answer generation requires hybrid retrieval"
                )
            if answer_enabled and answer_service_factory is None:
                raise ValueError(
                    "answer service factory is required when enabled"
                )
            if enabled:
                service = service_factory()
                service.prepare()
                app.state.hybrid_search_service = service
                if answer_enabled:
                    app.state.current_corpus_answer_service = (
                        answer_service_factory(service)  # type: ignore[misc]
                    )
            yield
        finally:
            if hasattr(app.state, "current_corpus_answer_service"):
                del app.state.current_corpus_answer_service
            if hasattr(app.state, "hybrid_search_service"):
                del app.state.hybrid_search_service

    return lifespan
