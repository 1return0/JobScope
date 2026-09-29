from __future__ import annotations

import argparse
import hashlib
import json
import time

from app.application.jobs.job_record_extraction_service import (
    JobRecordExtractionService,
)
from app.application.jobs.job_record_semantic_support import (
    JobRecordSemanticSupportEvaluator,
)
from app.config import load_settings
from app.domain.documents.document_chunking import (
    EvidenceChunk,
    build_evidence_id,
)
from app.domain.documents.document_ingestion import EvidenceLocation
from app.domain.documents.document_snapshot import build_snapshot_id
from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    StructuredJobRecordDraft,
)
from app.infrastructure.llm.openai_compatible_job_fact_support_judge import (
    OpenAiCompatibleJobFactSupportJudgeConfig,
    build_openai_compatible_job_fact_support_judge,
)
from app.infrastructure.llm.openai_compatible_job_record_model import (
    OpenAiCompatibleJobRecordModelConfig,
    build_openai_compatible_job_record_model,
)


_SOURCE_REFERENCE = "synthetic://job-record-extraction-smoke"
_EVIDENCE_TEXT = """示例科技2027校园招聘
岗位名称：AI Agent实习生
工作地点：上海
学历要求：本科及以上
专业要求：专业不限
申请截止日期：2026年9月30日
岗位职责：构建基于RAG的智能体工作流。
任职要求：熟悉Python和大模型应用开发。"""


def main() -> int:
    arguments = _parse_arguments()
    settings = load_settings(dotenv_override=True)
    if not settings.answer_generation_enabled:
        raise RuntimeError(
            "set JOBSCOPE_ANSWER_GENERATION_ENABLED=true before smoke test"
        )

    model_name = arguments.model_name or settings.answer_model_name
    model_config = OpenAiCompatibleJobRecordModelConfig(
        model_name=model_name,
        base_url=settings.answer_model_base_url,
        api_key_environment_variable=(
            settings.answer_model_api_key_environment_variable
        ),
        timeout_seconds=settings.answer_model_timeout_seconds,
        max_completion_tokens=max(
            settings.answer_model_max_completion_tokens,
            2400,
        ),
    )
    judge_config = OpenAiCompatibleJobFactSupportJudgeConfig(
        model_name=model_name,
        base_url=settings.answer_model_base_url,
        api_key_environment_variable=(
            settings.answer_model_api_key_environment_variable
        ),
        timeout_seconds=settings.answer_model_timeout_seconds,
    )
    service = JobRecordExtractionService(
        build_openai_compatible_job_record_model(model_config),
        JobRecordSemanticSupportEvaluator(
            build_openai_compatible_job_fact_support_judge(judge_config)
        ),
    )
    snapshot_id, chunk = _build_fixture()

    started_at = time.perf_counter()
    result = service.extract(
        source_snapshot_id=snapshot_id,
        chunks=(chunk,),
    )
    elapsed_seconds = time.perf_counter() - started_at
    print(
        json.dumps(
            {
                "status": result.status,
                "elapsed_seconds": elapsed_seconds,
                "source_snapshot_id": snapshot_id,
                "evidence_id": chunk.evidence_id,
                "model_identity": result.model_identity,
                "prompt_version": result.prompt_version,
                "extraction_contract_version": (
                    result.extraction_contract_version
                ),
                "record": _serialize_record(result.record),
                "grounding": {
                    "valid": result.grounding_report.valid,
                    "checked_fact_count": (
                        result.grounding_report.checked_fact_count
                    ),
                    "checked_citation_count": (
                        result.grounding_report.checked_citation_count
                    ),
                    "issue_codes": [
                        issue.code
                        for issue in result.grounding_report.issues
                    ],
                },
                "semantic_support": (
                    None
                    if result.semantic_support_report is None
                    else {
                        "valid": result.semantic_support_report.valid,
                        "supported_count": (
                            result.semantic_support_report.supported_count
                        ),
                        "unsupported_count": (
                            result.semantic_support_report.unsupported_count
                        ),
                        "uncertain_count": (
                            result.semantic_support_report.uncertain_count
                        ),
                        "judge_identity": (
                            result.semantic_support_report.judge_identity
                        ),
                    }
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real Qwen structured job extraction and support smoke."
        )
    )
    parser.add_argument("--model-name")
    return parser.parse_args()


def _build_fixture() -> tuple[str, EvidenceChunk]:
    content_sha256 = hashlib.sha256(
        _EVIDENCE_TEXT.encode("utf-8")
    ).hexdigest()
    location = EvidenceLocation(
        page_number=1,
        heading_path=("示例科技2027校园招聘",),
    )
    evidence_id = build_evidence_id(
        document_sha256=content_sha256,
        source_reference=_SOURCE_REFERENCE,
        source_fragment_ordinal=0,
        chunk_ordinal=0,
        chunker_version="structure-v1",
        text=_EVIDENCE_TEXT,
        location=location,
    )
    return (
        build_snapshot_id(
            source_reference=_SOURCE_REFERENCE,
            content_sha256=content_sha256,
        ),
        EvidenceChunk(
            evidence_id=evidence_id,
            document_sha256=content_sha256,
            source_reference=_SOURCE_REFERENCE,
            source_fragment_ordinal=0,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text=_EVIDENCE_TEXT,
            location=location,
        ),
    )


def _serialize_fact(fact: EvidenceBackedJobFact) -> object:
    if not fact.is_stated:
        return None
    return {
        "value": fact.value,
        "evidence_ids": [
            citation.evidence_id for citation in fact.citations
        ],
    }


def _serialize_record(record: StructuredJobRecordDraft) -> dict[str, object]:
    return {
        "company": _serialize_fact(record.company),
        "job_title": _serialize_fact(record.job_title),
        "locations": [_serialize_fact(item) for item in record.locations],
        "education_requirement": _serialize_fact(
            record.education_requirement
        ),
        "major_requirement": _serialize_fact(record.major_requirement),
        "recruitment_type": _serialize_fact(record.recruitment_type),
        "application_deadline": _serialize_fact(
            record.application_deadline
        ),
        "responsibilities": [
            _serialize_fact(item) for item in record.responsibilities
        ],
        "required_qualifications": [
            _serialize_fact(item)
            for item in record.required_qualifications
        ],
        "preferred_qualifications": [
            _serialize_fact(item)
            for item in record.preferred_qualifications
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
