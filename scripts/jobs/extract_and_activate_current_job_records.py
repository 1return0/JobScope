from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from sqlalchemy import select

from app.application.jobs.job_record_extraction_service import (
    JobRecordExtractionService,
)
from app.application.jobs.job_record_normalization import (
    ApplicationDeadlineNormalizer,
)
from app.application.jobs.job_record_semantic_support import (
    JobRecordSemanticSupportEvaluator,
)
from app.config import load_settings
from app.domain.documents.document_chunking import ChunkingConfig
from app.infrastructure.llm.openai_compatible_job_fact_support_judge import (
    OpenAiCompatibleJobFactSupportJudgeConfig,
    build_openai_compatible_job_fact_support_judge,
)
from app.infrastructure.llm.openai_compatible_job_record_model import (
    OpenAiCompatibleJobRecordModelConfig,
    build_openai_compatible_job_record_model,
)
from app.infrastructure.persistence.database import (
    create_postgresql_engine,
    create_session_factory,
)
from app.infrastructure.persistence.document_corpus_repository import (
    SqlAlchemyDocumentCorpusRepository,
)
from app.infrastructure.persistence.models import DocumentSnapshotRow
from app.infrastructure.persistence.structured_job_record_repository import (
    SqlAlchemyStructuredJobRecordRepository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, validate, persist and activate structured job records "
            "from current document snapshots."
        )
    )
    parser.add_argument(
        "--source-reference",
        action="append",
        required=True,
        help="Current snapshot source URL; repeat for multiple jobs.",
    )
    parser.add_argument(
        "--activation-reference",
        default="official-job-import",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    if not settings.answer_generation_enabled:
        raise RuntimeError(
            "set JOBSCOPE_ANSWER_GENERATION_ENABLED=true before extraction"
        )

    engine = create_postgresql_engine(settings)
    session_factory = create_session_factory(engine)
    corpus = SqlAlchemyDocumentCorpusRepository(session_factory)
    records = SqlAlchemyStructuredJobRecordRepository(session_factory)
    model_config = OpenAiCompatibleJobRecordModelConfig(
        model_name=settings.answer_model_name,
        base_url=settings.answer_model_base_url,
        api_key_environment_variable=(
            settings.answer_model_api_key_environment_variable
        ),
        timeout_seconds=settings.answer_model_timeout_seconds,
        max_completion_tokens=max(
            settings.answer_model_max_completion_tokens,
            2400,
        ),
        enable_thinking=False,
    )
    judge_config = OpenAiCompatibleJobFactSupportJudgeConfig(
        model_name=settings.answer_model_name,
        base_url=settings.answer_model_base_url,
        api_key_environment_variable=(
            settings.answer_model_api_key_environment_variable
        ),
        timeout_seconds=settings.answer_model_timeout_seconds,
        enable_thinking=False,
    )
    extraction = JobRecordExtractionService(
        build_openai_compatible_job_record_model(model_config),
        JobRecordSemanticSupportEvaluator(
            build_openai_compatible_job_fact_support_judge(judge_config)
        ),
    )
    deadline_normalizer = ApplicationDeadlineNormalizer()
    reports: list[dict[str, object]] = []
    failed = False

    try:
        for source_reference in args.source_reference:
            normalized_source = source_reference.strip()
            snapshot_id = _current_snapshot_id(
                session_factory,
                source_reference=normalized_source,
            )
            chunks = tuple(
                corpus.list_current_chunks(
                    chunker_version=ChunkingConfig().identity,
                    source_reference=normalized_source,
                )
            )
            result = extraction.extract(
                source_snapshot_id=snapshot_id,
                chunks=chunks,
            )
            report: dict[str, object] = {
                "source_reference": normalized_source,
                "source_snapshot_id": snapshot_id,
                "chunk_count": len(chunks),
                "extraction_status": result.status,
                "grounding_valid": result.grounding_report.valid,
                "semantic_support_valid": (
                    result.semantic_support_report.valid
                    if result.semantic_support_report is not None
                    else None
                ),
                "record_id": None,
                "created": False,
                "activated": False,
            }
            if result.status != "accepted":
                failed = True
                report["grounding_issue_codes"] = [
                    issue.code for issue in result.grounding_report.issues
                ]
                if result.semantic_support_report is not None:
                    report["semantic_counts"] = {
                        "supported": (
                            result.semantic_support_report.supported_count
                        ),
                        "unsupported": (
                            result.semantic_support_report.unsupported_count
                        ),
                        "uncertain": (
                            result.semantic_support_report.uncertain_count
                        ),
                    }
                    report["semantic_review_items"] = [
                        {
                            "field_path": item.field_path,
                            "value": item.value,
                            "label": item.decision.label,
                            "rationale": item.decision.rationale,
                        }
                        for item in result.semantic_support_report.evaluations
                        if item.decision.label != "supported"
                    ]
                reports.append(report)
                continue

            deadline = deadline_normalizer.normalize(
                result.record.application_deadline
            )
            persisted = records.persist_accepted(
                result,
                deadline=deadline,
            )
            activated = records.activate(
                persisted.record_id,
                activated_at=datetime.now(UTC),
                activation_reference=(
                    f"{args.activation_reference}:{normalized_source}"
                ),
            )
            report.update(
                {
                    "record_id": persisted.record_id,
                    "created": persisted.created,
                    "citations_inserted": persisted.citations_inserted,
                    "activated": activated.changed,
                    "deadline_status": deadline.status,
                    "deadline_reason_code": deadline.reason_code,
                }
            )
            reports.append(report)
    finally:
        engine.dispose()

    print(
        json.dumps(
            {
                "status": "completed_with_review" if failed else "completed",
                "total": len(reports),
                "accepted": sum(
                    item["extraction_status"] == "accepted"
                    for item in reports
                ),
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2 if failed else 0


def _current_snapshot_id(
    session_factory,
    *,
    source_reference: str,
) -> str:
    with session_factory() as session:
        snapshot_id = session.scalar(
            select(DocumentSnapshotRow.snapshot_id).where(
                DocumentSnapshotRow.source_reference == source_reference,
                DocumentSnapshotRow.is_current.is_(True),
            )
        )
    if snapshot_id is None:
        raise LookupError(
            f"current document snapshot not found: {source_reference}"
        )
    return snapshot_id


if __name__ == "__main__":
    raise SystemExit(main())
