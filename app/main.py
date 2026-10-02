from datetime import date
from typing import Literal, TypeAlias

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from app.api.documents.document_api import build_document_router
from app.api.evaluation.evaluation_api import build_evaluation_router
from app.api.jobs.job_agent_api import build_job_agent_router
from app.api.answering.answer_api import build_answer_router
from app.api.retrieval.retrieval_api import build_retrieval_router
from app.api.application_lifecycle import build_application_lifespan
from app.bootstrap import (
    build_document_parsing_service,
    build_document_processing_service,
    build_verified_domain_registry,
)
from app.config import load_settings
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchService,
)
from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerService,
)
from app.runtime_composition import (
    compose_answer_service,
    compose_hybrid_search_service,
    compose_job_agent_runtime,
)


settings = load_settings()
verified_domain_registry = build_verified_domain_registry(settings)
document_parsing_service = build_document_parsing_service(settings)
document_processing_service = build_document_processing_service(
    settings,
    parsing_service=document_parsing_service,
)


def build_hybrid_search_service(
    *,
    max_concurrent_queries: int | None = None,
):
    return compose_hybrid_search_service(
        settings,
        max_concurrent_queries=(
            max_concurrent_queries
        ),
    )


def build_answer_service(
    hybrid_search_service: CurrentCorpusHybridSearchService,
) -> CurrentCorpusAnswerService:
    return compose_answer_service(
        settings,
        hybrid_search_service,
    )


def build_job_agent_runtime(
    hybrid_search_service: CurrentCorpusHybridSearchService | None,
    answer_service: CurrentCorpusAnswerService | None,
):
    # The Answer service already owns the prepared Hybrid service. Passing the
    # lifecycle-managed instance prevents a second model load and index build.
    return compose_job_agent_runtime(
        settings,
        answer_service=answer_service,
    )


app = FastAPI(
    title=settings.service_name,
    version=settings.version,
    description=(
        "A multi-format, version-aware and evidence-grounded campus recruitment "
        "intelligence service."
    ),
    lifespan=build_application_lifespan(
        retrieval_enabled=settings.hybrid_retrieval_enabled,
        retrieval_service_factory=build_hybrid_search_service,
        answer_enabled=settings.answer_generation_enabled,
        answer_service_factory=build_answer_service,
        job_agent_enabled=settings.job_agent_enabled,
        job_agent_runtime_factory=build_job_agent_runtime,
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)
app.include_router(
    build_document_router(
        document_parsing_service,
        document_processing_service,
        max_upload_bytes=settings.max_upload_bytes,
    )
)
app.include_router(build_retrieval_router())
app.include_router(build_answer_router())
app.include_router(build_job_agent_router())
app.include_router(build_evaluation_router())


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ServiceMetaResponse(BaseModel):
    stage: str
    domain: str
    supported_formats: list[str]
    implemented_capabilities: list[str]
    planned_capabilities: list[str]
    read_only: bool


SourceKind: TypeAlias = Literal[
    "official_company",
    "official_university",
    "job_board",
]


class SourceCandidateRequest(BaseModel):
    company: str = Field(min_length=1, max_length=100)
    job_title: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl
    source_kind: SourceKind


class SourceCandidateValidationResponse(BaseModel):
    schema_valid: bool
    normalized_url: str


class SourceCandidateAssessmentResponse(BaseModel):
    source_kind: SourceKind
    domain_verified: bool
    transport_secure: bool
    content_freshness_verified: bool
    trust_verified: bool
    manual_review_required: bool
    review_reasons: list[str]
    matched_official_domain: str | None
    domain_evidence_reference: str | None
    domain_verified_at: date | None


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="UP",
        service=settings.service_name,
        version=settings.version,
        environment=settings.environment,
    )


@app.get(
    "/v1/meta",
    response_model=ServiceMetaResponse,
    tags=["system"],
    summary="Get JobScope service metadata",
    description=(
        "Returns the current development stage, supported input formats, "
        "implemented capabilities and planned capabilities."
    ),
)
def service_meta() -> ServiceMetaResponse:
    return ServiceMetaResponse(
        stage="productization",
        domain="candidate-side campus recruitment intelligence",
        supported_formats=["html", "pdf", "docx", "xlsx", "pptx"],
        implemented_capabilities=[
            "health-check",
            "typed-service-metadata",
            "source-schema-validation",
            "conservative-source-assessment",
            "verified-domain-matching",
            "verified-domain-csv-loading",
            "postgresql-domain-repository",
            "alembic-schema-migrations",
            "idempotent-domain-import",
            "unified-document-ingestion-contract",
            "html-document-parsing",
            "docx-document-parsing",
            "pdf-text-layer-parsing",
            "xlsx-row-parsing",
            "pptx-slide-parsing",
            "default-multi-format-parser-composition",
            "bounded-temporary-document-upload-api",
            "stable-evidence-id-contract",
            "structure-preserving-character-chunking",
            "parse-and-chunk-document-processing-preview",
            "postgresql-evidence-chunk-repository",
            "versioned-document-snapshot-repository",
            "atomic-document-corpus-persistence",
            "current-version-corpus-read-model",
            "in-memory-bm25-lexical-baseline",
            "retrieval-evaluation-metrics-contract",
            "document-parsing-failure-classification",
            "verified-corpus-manifest-loading",
            "immutable-corpus-manifest-freezing",
            "manifest-driven-corpus-ingestion",
            "scanned-pdf-ocr",
            "hybrid-retrieval",
            "evidence-grounded-answers",
            "structured-job-record-extraction",
            "current-job-record-query",
            "job-agent-planning",
            "registered-tool-execution",
            "planner-evaluation-and-quality-gates",
            "fastapi-job-agent-runtime-lifecycle",
            "recruitment-type-normalization",
            "model-provider-error-translation",
            "frontend-operational-workbench",
            "evaluation-artifact-catalog-api",
        ],
        planned_capabilities=[
            "human-in-the-loop-review-ui",
            "persistent-multi-worker-index",
            "production-load-validation",
            "multi-agent-orchestration",
        ],
        read_only=True,
    )


@app.post(
    "/v1/source-candidates/validate",
    response_model=SourceCandidateValidationResponse,
    tags=["sources"],
    summary="Validate a recruitment source candidate",
    description=(
        "Validates the request schema without downloading or persisting "
        "the recruitment source."
    ),
)
def validate_source_candidate(
    candidate: SourceCandidateRequest,
) -> SourceCandidateValidationResponse:
    return SourceCandidateValidationResponse(
        schema_valid=True,
        normalized_url=str(candidate.source_url),
    )


@app.post(
    "/v1/source-candidates/assess",
    response_model=SourceCandidateAssessmentResponse,
    tags=["sources"],
    summary="Assess a recruitment source candidate",
    description=(
        "Applies conservative business rules after schema validation. "
        "This endpoint does not claim that a source is authentic."
    ),
)
def assess_source_candidate(
    candidate: SourceCandidateRequest,
) -> SourceCandidateAssessmentResponse:
    matched_official_domain: str | None = None
    domain_evidence_reference: str | None = None
    domain_verified_at: date | None = None
    review_reasons: list[str] = []

    if candidate.source_kind == "official_company":
        matched_record = verified_domain_registry.find_matching_record(
            company=candidate.company,
            candidate_host=candidate.source_url.host or "",
        )
        if matched_record is None:
            review_reasons.append(
                "company-domain-ownership-not-verified"
            )
        else:
            matched_official_domain = matched_record.domain
            domain_evidence_reference = (
                matched_record.evidence_reference
            )
            domain_verified_at = matched_record.verified_at
    elif candidate.source_kind == "official_university":
        review_reasons.append("secondary-source-requires-cross-check")
    else:
        review_reasons.append("aggregated-source-requires-cross-check")

    if candidate.source_url.scheme != "https":
        review_reasons.append("non-https-url")

    domain_verified = (
        candidate.source_kind == "official_company"
        and matched_official_domain is not None
    )
    transport_secure = candidate.source_url.scheme == "https"
    content_freshness_verified = False
    review_reasons.append("content-freshness-not-verified")

    trust_verified = (
        domain_verified
        and transport_secure
        and content_freshness_verified
    )

    return SourceCandidateAssessmentResponse(
        source_kind=candidate.source_kind,
        domain_verified=domain_verified,
        transport_secure=transport_secure,
        content_freshness_verified=content_freshness_verified,
        trust_verified=trust_verified,
        manual_review_required=not trust_verified,
        review_reasons=review_reasons,
        matched_official_domain=matched_official_domain,
        domain_evidence_reference=domain_evidence_reference,
        domain_verified_at=domain_verified_at,
    )
