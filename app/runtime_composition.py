from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine

from app.application.jobs.job_agent_graph import build_job_agent_graph
from app.application.jobs.job_agent_planning import JobAgentPlanner
from app.application.jobs.job_search_tool import SearchCurrentJobsTool
from app.application.jobs.job_rag_tools import (
    AnswerRecruitmentQuestionTool,
    SearchJobsWithEvidenceTool,
)
from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerService,
)
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchService,
)
from app.application.answering.grounded_answer_service import GroundedAnswerService
from app.bootstrap import build_current_corpus_hybrid_search_service
from app.config import Settings
from app.infrastructure.retrieval.embedding_model_presets import (
    EMBEDDING_MODEL_PRESETS,
)
from app.infrastructure.llm.openai_compatible_answer_model import (
    OpenAiCompatibleAnswerModelConfig,
    build_openai_compatible_answer_model,
)
from app.infrastructure.llm.openai_compatible_job_agent_planner import (
    OpenAiCompatibleJobAgentPlannerConfig,
    build_openai_compatible_job_agent_planner,
)
from app.infrastructure.persistence.database import (
    create_postgresql_engine,
    create_session_factory,
)
from app.infrastructure.persistence.structured_job_record_query import (
    SqlAlchemyCurrentJobRecordQuery,
)
from app.infrastructure.retrieval.sentence_transformer_embedder import (
    SentenceTransformerTextEmbedder,
)


@dataclass(slots=True)
class JobAgentRuntime:
    graph: Any
    engine: Engine

    def invoke(self, user_query: str) -> dict[str, Any]:
        return self.graph.invoke({"user_query": user_query})

    def close(self) -> None:
        self.engine.dispose()


def compose_hybrid_search_service(
    settings: Settings,
    *,
    max_concurrent_queries: int | None = None,
) -> CurrentCorpusHybridSearchService:
    try:
        embedding_spec = EMBEDDING_MODEL_PRESETS[
            settings.embedding_preset
        ]
    except KeyError as error:
        raise ValueError(
            "unsupported JOBSCOPE_EMBEDDING_PRESET"
        ) from error
    embedder = SentenceTransformerTextEmbedder(
        embedding_spec,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        cache_folder=settings.embedding_cache_folder,
        local_files_only=True,
    )
    return build_current_corpus_hybrid_search_service(
        settings,
        embedder,
        candidate_k=settings.hybrid_candidate_k,
        rank_constant=settings.hybrid_rank_constant,
        max_concurrent_queries=(
            settings.hybrid_max_concurrent_queries
            if max_concurrent_queries is None
            else max_concurrent_queries
        ),
    )


def compose_answer_service(
    settings: Settings,
    hybrid_search_service: CurrentCorpusHybridSearchService,
) -> CurrentCorpusAnswerService:
    model = build_openai_compatible_answer_model(
        OpenAiCompatibleAnswerModelConfig(
            model_name=settings.answer_model_name,
            base_url=settings.answer_model_base_url,
            api_key_environment_variable=(
                settings.answer_model_api_key_environment_variable
            ),
            timeout_seconds=settings.answer_model_timeout_seconds,
            max_completion_tokens=(
                settings.answer_model_max_completion_tokens
            ),
        )
    )
    return CurrentCorpusAnswerService(
        hybrid_search_service,
        GroundedAnswerService(model),
    )


def compose_job_agent_runtime(
    settings: Settings,
    *,
    planner: JobAgentPlanner | None = None,
    engine: Engine | None = None,
    answer_service: CurrentCorpusAnswerService | None = None,
) -> JobAgentRuntime:
    selected_engine = engine or create_postgresql_engine(settings)
    selected_planner = planner or build_openai_compatible_job_agent_planner(
        OpenAiCompatibleJobAgentPlannerConfig(
            model_name=settings.answer_model_name,
            base_url=settings.answer_model_base_url,
            api_key_environment_variable=(
                settings.answer_model_api_key_environment_variable
            ),
            timeout_seconds=settings.answer_model_timeout_seconds,
            max_completion_tokens=min(
                settings.answer_model_max_completion_tokens,
                800,
            ),
            enable_thinking=False,
        )
    )
    session_factory = create_session_factory(selected_engine)
    current_jobs = SqlAlchemyCurrentJobRecordQuery(session_factory)
    search_tool = SearchCurrentJobsTool(current_jobs)
    tools = [search_tool]
    if answer_service is not None:
        answer_tool = AnswerRecruitmentQuestionTool(answer_service)
        tools.extend(
            [
                answer_tool,
                SearchJobsWithEvidenceTool(search_tool, answer_tool),
            ]
        )
    return JobAgentRuntime(
        graph=build_job_agent_graph(selected_planner, tools),
        engine=selected_engine,
    )
