from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.application.jobs.job_record_extraction_service import (
    JobRecordExtractionResult,
)
from app.application.jobs.job_record_normalization import (
    NormalizedApplicationDeadline,
)
from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    StructuredJobRecordDraft,
)
from app.domain.jobs.recruitment_type import normalize_recruitment_type
from app.infrastructure.persistence.models import (
    DocumentSnapshotChunkRow,
    DocumentSnapshotRow,
    StructuredJobRecordCitationRow,
    StructuredJobRecordRow,
)


class UnacceptedJobRecordError(ValueError):
    pass


class JobRecordPersistenceEvidenceError(ValueError):
    pass


class StructuredJobRecordNotFoundError(LookupError):
    pass


class StaleJobRecordActivationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PersistStructuredJobRecordReport:
    record_id: str
    created: bool
    citations_inserted: int


@dataclass(frozen=True, slots=True)
class ActivateStructuredJobRecordReport:
    record_id: str
    source_snapshot_id: str
    previous_record_id: str | None
    changed: bool


class SqlAlchemyStructuredJobRecordRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def persist_accepted(
        self,
        result: JobRecordExtractionResult,
        *,
        deadline: NormalizedApplicationDeadline,
    ) -> PersistStructuredJobRecordReport:
        self._validate_accepted_result(result, deadline=deadline)
        payload = _record_payload(result.record)
        fingerprint = _record_fingerprint(
            result,
            payload=payload,
            deadline=deadline,
        )
        record_id = f"job_{fingerprint}"
        citations = tuple(_iter_citations(result.record))

        with self._session_factory.begin() as session:
            existing_id = session.scalar(
                select(StructuredJobRecordRow.record_id).where(
                    StructuredJobRecordRow.record_fingerprint
                    == fingerprint
                )
            )
            if existing_id is not None:
                return PersistStructuredJobRecordReport(
                    record_id=existing_id,
                    created=False,
                    citations_inserted=0,
                )

            linked_evidence_ids = set(
                session.scalars(
                    select(
                        DocumentSnapshotChunkRow.evidence_id
                    ).where(
                        DocumentSnapshotChunkRow.snapshot_id
                        == result.record.source_snapshot_id
                    )
                ).all()
            )
            cited_evidence_ids = {
                citation.evidence_id
                for _, _, citation in citations
            }
            if not cited_evidence_ids.issubset(linked_evidence_ids):
                raise JobRecordPersistenceEvidenceError(
                    "every citation must belong to the source snapshot"
                )

            record = result.record
            semantic_report = result.semantic_support_report
            if semantic_report is None:
                raise AssertionError(
                    "accepted result requires semantic support report"
                )
            session.add(
                StructuredJobRecordRow(
                    record_id=record_id,
                    record_fingerprint=fingerprint,
                    source_snapshot_id=record.source_snapshot_id,
                    extraction_contract_version=(
                        result.extraction_contract_version
                    ),
                    prompt_version=result.prompt_version,
                    model_identity=result.model_identity,
                    judge_identity=semantic_report.judge_identity,
                    record_payload=payload,
                    company=record.company.value,
                    job_title=record.job_title.value,
                    locations=[
                        fact.value
                        for fact in record.locations
                        if fact.value is not None
                    ],
                    education_requirement=(
                        record.education_requirement.value
                    ),
                    major_requirement=record.major_requirement.value,
                    recruitment_type=(
                        normalize_recruitment_type(
                            record.recruitment_type.value
                        )
                        if record.recruitment_type.value is not None
                        else None
                    ),
                    application_deadline_raw=(
                        record.application_deadline.value
                    ),
                    application_deadline_normalized=(
                        deadline.normalized_date
                    ),
                    deadline_normalization_status=deadline.status,
                    deadline_reason_code=deadline.reason_code,
                    date_normalizer_version=deadline.normalizer_version,
                )
            )
            session.flush()
            session.add_all(
                StructuredJobRecordCitationRow(
                    record_id=record_id,
                    field_path=field_path,
                    citation_ordinal=citation_ordinal,
                    evidence_id=citation.evidence_id,
                    quoted_text=citation.quoted_text,
                )
                for field_path, citation_ordinal, citation in citations
            )
            session.flush()

        return PersistStructuredJobRecordReport(
            record_id=record_id,
            created=True,
            citations_inserted=len(citations),
        )

    def activate(
        self,
        record_id: str,
        *,
        activated_at: datetime,
        activation_reference: str,
    ) -> ActivateStructuredJobRecordReport:
        normalized_id = record_id.strip()
        normalized_reference = activation_reference.strip()
        if not normalized_id:
            raise ValueError("record_id must not be blank")
        if not normalized_reference:
            raise ValueError("activation_reference must not be blank")
        if (
            activated_at.tzinfo is None
            or activated_at.utcoffset() is None
        ):
            raise ValueError("activated_at must be timezone-aware")

        with self._session_factory.begin() as session:
            source_snapshot_id = session.scalar(
                select(StructuredJobRecordRow.source_snapshot_id).where(
                    StructuredJobRecordRow.record_id == normalized_id
                )
            )
            if source_snapshot_id is None:
                raise StructuredJobRecordNotFoundError(
                    f"structured job record not found: {normalized_id}"
                )
            # Every activation for the same source snapshot locks this one
            # common row first. A stable lock order avoids two tasks first
            # locking different job records and then deadlocking while each
            # waits for the other's sibling record.
            source_snapshot = session.scalar(
                select(DocumentSnapshotRow)
                .where(
                    DocumentSnapshotRow.snapshot_id
                    == source_snapshot_id
                )
                .with_for_update()
            )
            if source_snapshot is None or not source_snapshot.is_current:
                raise StaleJobRecordActivationError(
                    "job record source snapshot is not current"
                )

            sibling_rows = session.scalars(
                select(StructuredJobRecordRow)
                .where(
                    StructuredJobRecordRow.source_snapshot_id
                    == source_snapshot_id
                )
                .order_by(StructuredJobRecordRow.record_id)
                .with_for_update()
            ).all()
            selected = next(
                (
                    row
                    for row in sibling_rows
                    if row.record_id == normalized_id
                ),
                None,
            )
            if selected is None:
                raise StructuredJobRecordNotFoundError(
                    f"structured job record not found: {normalized_id}"
                )
            previous = next(
                (row for row in sibling_rows if row.is_current),
                None,
            )
            if previous is not None and previous.record_id == normalized_id:
                return ActivateStructuredJobRecordReport(
                    record_id=selected.record_id,
                    source_snapshot_id=selected.source_snapshot_id,
                    previous_record_id=previous.record_id,
                    changed=False,
                )

            for row in sibling_rows:
                row.is_current = False
            selected.is_current = True
            selected.activated_at = activated_at
            selected.activation_reference = normalized_reference
            session.flush()

        return ActivateStructuredJobRecordReport(
            record_id=selected.record_id,
            source_snapshot_id=selected.source_snapshot_id,
            previous_record_id=(
                previous.record_id if previous is not None else None
            ),
            changed=True,
        )

    @staticmethod
    def _validate_accepted_result(
        result: JobRecordExtractionResult,
        *,
        deadline: NormalizedApplicationDeadline,
    ) -> None:
        semantic_report = result.semantic_support_report
        if (
            result.status != "accepted"
            or not result.grounding_report.valid
            or semantic_report is None
            or not semantic_report.valid
        ):
            raise UnacceptedJobRecordError(
                "only fully accepted job records may be persisted"
            )
        if not result.record.job_title.is_stated:
            raise UnacceptedJobRecordError(
                "accepted job record must contain an evidence-backed job title"
            )
        if deadline.raw_fact != result.record.application_deadline:
            raise ValueError(
                "normalized deadline must belong to the persisted record"
            )


def _iter_facts(record: StructuredJobRecordDraft):
    for field_name in (
        "company",
        "job_title",
        "education_requirement",
        "major_requirement",
        "recruitment_type",
        "application_deadline",
    ):
        yield field_name, getattr(record, field_name)
    for field_name in (
        "locations",
        "responsibilities",
        "required_qualifications",
        "preferred_qualifications",
    ):
        for index, fact in enumerate(getattr(record, field_name)):
            yield f"{field_name}[{index}]", fact


def _iter_citations(record: StructuredJobRecordDraft):
    for field_path, fact in _iter_facts(record):
        for citation_ordinal, citation in enumerate(
            fact.citations,
            start=1,
        ):
            yield field_path, citation_ordinal, citation


def _fact_payload(fact: EvidenceBackedJobFact) -> dict:
    return {
        "value": fact.value,
        "citations": [
            {
                "evidence_id": citation.evidence_id,
                "quoted_text": citation.quoted_text,
            }
            for citation in fact.citations
        ],
    }


def _record_payload(record: StructuredJobRecordDraft) -> dict:
    return {
        "source_snapshot_id": record.source_snapshot_id,
        "extraction_contract_version": (
            record.extraction_contract_version
        ),
        "company": _fact_payload(record.company),
        "job_title": _fact_payload(record.job_title),
        "locations": [_fact_payload(fact) for fact in record.locations],
        "education_requirement": _fact_payload(
            record.education_requirement
        ),
        "major_requirement": _fact_payload(record.major_requirement),
        "recruitment_type": _fact_payload(record.recruitment_type),
        "application_deadline": _fact_payload(
            record.application_deadline
        ),
        "responsibilities": [
            _fact_payload(fact) for fact in record.responsibilities
        ],
        "required_qualifications": [
            _fact_payload(fact)
            for fact in record.required_qualifications
        ],
        "preferred_qualifications": [
            _fact_payload(fact)
            for fact in record.preferred_qualifications
        ],
    }


def _record_fingerprint(
    result: JobRecordExtractionResult,
    *,
    payload: dict,
    deadline: NormalizedApplicationDeadline,
) -> str:
    semantic_report = result.semantic_support_report
    if semantic_report is None:
        raise AssertionError("accepted result requires semantic report")
    identity = {
        "record": payload,
        "prompt_version": result.prompt_version,
        "model_identity": result.model_identity,
        "judge_identity": semantic_report.judge_identity,
        "deadline": {
            "normalized_date": (
                deadline.normalized_date.isoformat()
                if deadline.normalized_date is not None
                else None
            ),
            "status": deadline.status,
            "reason_code": deadline.reason_code,
            "normalizer_version": deadline.normalizer_version,
        },
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
