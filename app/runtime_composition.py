from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import Engine

from app.application.jobs.job_agent_graph import build_job_agent_graph
from app.application.jobs.agent_harness import (
    JobAgentHarness,
    JobAgentHarnessLimits,
    JobAgentHarnessSnapshot,
)
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
from app.infrastructure.llm.retrying_job_agent_planner import (
    JobAgentPlannerRetryPolicy,
    RetryingJobAgentPlanner,
)
from app.infrastructure.persistence.database import (
    create_postgresql_engine,
    create_session_factory,
)
from app.infrastructure.persistence.structured_job_record_query import (
    SqlAlchemyCurrentJobRecordQuery,
)
from app.infrastructure.persistence.job_agent_session_repository import (
    SqlAlchemyJobAgentSessionStore,
)
from app.application.jobs.job_agent_sessions import (
    IssuedJobAgentSession,
    JobAgentSessionService,
)
from app.application.jobs.job_agent_thread_identity import (
    build_job_agent_thread_id,
)
from app.application.memory.long_term_memory import (
    LongTermMemoryEntry,
    LongTermMemoryService,
)
from app.infrastructure.persistence.database import build_postgresql_dsn
from app.infrastructure.postgres_checkpointer import open_postgres_checkpointer
from app.infrastructure.postgres_long_term_memory_store import (
    open_postgres_long_term_memory_store,
)
from app.infrastructure.retrieval.sentence_transformer_embedder import (
    SentenceTransformerTextEmbedder,
)


@dataclass(slots=True)
class JobAgentRuntime:
    graph: Any
    engine: Engine
    session_service: JobAgentSessionService
    session_cookie_secure: bool
    harness_limits: JobAgentHarnessLimits = field(default_factory=JobAgentHarnessLimits)
    memory_service: LongTermMemoryService | None = None
    resource_closer: Any | None = None

    def invoke(self, user_query: str, *, owner_id: str) -> dict[str, Any]:
        if not owner_id.strip():
            raise ValueError("job agent owner_id must not be blank")
        preferences = self.list_preferences(owner_id=owner_id)
        harness = JobAgentHarness(self.harness_limits)
        state = self.graph.invoke(
            {"user_query": user_query, "owner_id": owner_id},
            config={
                "configurable": {
                    "thread_id": build_job_agent_thread_id(owner_id),
                }
            },
            context={
                "preferences": {
                    entry.preference_key: entry.preference_value
                    for entry in preferences
                },
                "harness": harness,
            },
        )
        state["harness_snapshot"] = harness.snapshot()
        return state

    def issue_browser_session(self) -> IssuedJobAgentSession:
        return self.session_service.issue_session()

    def resolve_browser_owner(self, session_token: str | None) -> str:
        return self.session_service.resolve_owner_id(session_token)

    def get_conversation_turns(
        self,
        *,
        owner_id: str,
    ) -> tuple[dict[str, str], ...]:
        """Expose only the bounded query history for the trusted owner."""
        snapshot = self.graph.get_state(
            {
                "configurable": {
                    "thread_id": build_job_agent_thread_id(owner_id),
                }
            }
        )
        values = getattr(snapshot, "values", {}) if snapshot is not None else {}
        turns = values.get("conversation_turns", ())
        return tuple(
            turn
            for turn in turns
            if isinstance(turn, dict)
            and isinstance(turn.get("role"), str)
            and isinstance(turn.get("content"), str)
        )

    def clear_conversation_turns(self, *, owner_id: str) -> bool:
        checkpointer = getattr(self.graph, "checkpointer", None)
        delete_thread = getattr(checkpointer, "delete_thread", None)
        if not callable(delete_thread):
            raise RuntimeError("Job Agent conversation memory is not prepared")
        thread_id = build_job_agent_thread_id(owner_id)
        existing = self.graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if existing is None:
            return False
        delete_thread(thread_id)
        return True

    def remember_preference(
        self,
        *,
        owner_id: str,
        preference_key: str,
        preference_value: str,
        user_consented: bool,
    ) -> LongTermMemoryEntry:
        return self._require_memory_service().remember_preference(
            owner_id=owner_id,
            preference_key=preference_key,
            preference_value=preference_value,
            user_consented=user_consented,
        )

    def list_preferences(
        self,
        *,
        owner_id: str,
    ) -> tuple[LongTermMemoryEntry, ...]:
        if self.memory_service is None:
            return ()
        return self.memory_service.list_preferences(owner_id=owner_id)

    def forget_preference(
        self,
        *,
        owner_id: str,
        preference_key: str,
    ) -> bool:
        return self._require_memory_service().forget_preference(
            owner_id=owner_id,
            preference_key=preference_key,
        )

    def forget_all_preferences(self, *, owner_id: str) -> int:
        return self._require_memory_service().forget_all_preferences(
            owner_id=owner_id
        )

    def _require_memory_service(self) -> LongTermMemoryService:
        if self.memory_service is None:
            raise RuntimeError("Job Agent long-term memory is not prepared")
        return self.memory_service

    def close(self) -> None:
        if self.resource_closer is not None:
            self.resource_closer()
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
    checkpointing_enabled: bool = True,
    checkpointer: Any | None = None,
    memory_service: LongTermMemoryService | None = None,
) -> JobAgentRuntime:
    selected_engine = engine or create_postgresql_engine(settings)
    resources = ExitStack()
    selected_planner = planner
    if selected_planner is None:
        model_planner = build_openai_compatible_job_agent_planner(
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
        from app.application.jobs.agent_harness import HarnessedJobAgentPlanner

        selected_planner = RetryingJobAgentPlanner(
            HarnessedJobAgentPlanner(model_planner),
            JobAgentPlannerRetryPolicy(
                max_attempts=settings.job_agent_planner_max_attempts,
                initial_backoff_seconds=(
                    settings.job_agent_planner_initial_backoff_seconds
                ),
            ),
        )
    session_factory = create_session_factory(selected_engine)
    session_service = JobAgentSessionService(
        SqlAlchemyJobAgentSessionStore(session_factory),
        ttl=timedelta(seconds=settings.job_agent_session_ttl_seconds),
    )
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
    try:
        selected_memory_service = memory_service
        if selected_memory_service is None and engine is None:
            memory_store = resources.enter_context(
                open_postgres_long_term_memory_store(
                    build_postgresql_dsn(settings),
                    application_namespace="jobscope",
                )
            )
            selected_memory_service = LongTermMemoryService(
                memory_store,
                allowed_preference_keys=frozenset(
                    {"preferred_location", "preferred_recruitment_type"}
                ),
            )
        selected_checkpointer = checkpointer
        if selected_checkpointer is None and checkpointing_enabled:
            selected_checkpointer = resources.enter_context(
                open_postgres_checkpointer(build_postgresql_dsn(settings))
            )
        return JobAgentRuntime(
            graph=build_job_agent_graph(
                selected_planner,
                tools,
                checkpointer=selected_checkpointer,
            ),
            engine=selected_engine,
            session_service=session_service,
            session_cookie_secure=settings.job_agent_session_cookie_secure,
            harness_limits=JobAgentHarnessLimits(
                max_model_calls=settings.job_agent_harness_max_model_calls,
                max_tool_calls=settings.job_agent_harness_max_tool_calls,
                max_elapsed_seconds=settings.job_agent_harness_max_elapsed_seconds,
            ),
            memory_service=selected_memory_service,
            resource_closer=resources.close,
        )
    except Exception:
        resources.close()
        if engine is None:
            selected_engine.dispose()
        raise
