from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.retrieval.retrieval_lifecycle import (
    AnswerServiceFactory,
    HybridSearchServiceFactory,
    build_retrieval_lifespan,
)
from app.runtime_composition import JobAgentRuntime
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchService,
)
from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerService,
)


JobAgentRuntimeFactory = Callable[
    [
        CurrentCorpusHybridSearchService | None,
        CurrentCorpusAnswerService | None,
    ],
    JobAgentRuntime,
]


def build_application_lifespan(
    *,
    retrieval_enabled: bool,
    retrieval_service_factory: HybridSearchServiceFactory,
    answer_enabled: bool = False,
    answer_service_factory: AnswerServiceFactory | None = None,
    job_agent_enabled: bool = False,
    job_agent_runtime_factory: JobAgentRuntimeFactory | None = None,
):
    retrieval_lifespan = build_retrieval_lifespan(
        enabled=retrieval_enabled,
        service_factory=retrieval_service_factory,
        answer_enabled=answer_enabled,
        answer_service_factory=answer_service_factory,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if job_agent_enabled and job_agent_runtime_factory is None:
            raise ValueError(
                "Job Agent runtime factory is required when enabled"
            )
        runtime: JobAgentRuntime | None = None
        async with retrieval_lifespan(app):
            try:
                if job_agent_enabled:
                    runtime = job_agent_runtime_factory(  # type: ignore[misc]
                        getattr(app.state, "hybrid_search_service", None),
                        getattr(
                            app.state,
                            "current_corpus_answer_service",
                            None,
                        ),
                    )
                    app.state.job_agent_runtime = runtime
                yield
            finally:
                if hasattr(app.state, "job_agent_runtime"):
                    del app.state.job_agent_runtime
                if runtime is not None:
                    runtime.close()

    return lifespan
