from __future__ import annotations

import json
from typing import Any

from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    JobFieldCitation,
    StructuredJobRecordDraft,
)


class JobRecordOutputDecodingError(ValueError):
    pass


class JobRecordOutputDecoder:
    def decode(
        self,
        raw_output: str,
        *,
        source_snapshot_id: str,
        extraction_contract_version: str,
    ) -> StructuredJobRecordDraft:
        normalized_snapshot_id = source_snapshot_id.strip()
        normalized_contract_version = extraction_contract_version.strip()
        if not normalized_snapshot_id:
            raise ValueError("source_snapshot_id must not be blank")
        if not normalized_contract_version:
            raise ValueError(
                "extraction_contract_version must not be blank"
            )

        try:
            payload = json.loads(raw_output)
            item = _object(payload, "job record")
            _require_exact_fields(
                item,
                {
                    "company",
                    "job_title",
                    "locations",
                    "education_requirement",
                    "major_requirement",
                    "recruitment_type",
                    "application_deadline",
                    "responsibilities",
                    "required_qualifications",
                    "preferred_qualifications",
                },
                field_name="job record",
            )
            return StructuredJobRecordDraft(
                source_snapshot_id=normalized_snapshot_id,
                extraction_contract_version=normalized_contract_version,
                company=_fact(item, "company"),
                job_title=_fact(item, "job_title"),
                locations=_facts(item, "locations"),
                education_requirement=_fact(
                    item,
                    "education_requirement",
                ),
                major_requirement=_fact(item, "major_requirement"),
                recruitment_type=_fact(item, "recruitment_type"),
                application_deadline=_fact(
                    item,
                    "application_deadline",
                ),
                responsibilities=_facts(item, "responsibilities"),
                required_qualifications=_facts(
                    item,
                    "required_qualifications",
                ),
                preferred_qualifications=_facts(
                    item,
                    "preferred_qualifications",
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise JobRecordOutputDecodingError(
                "model returned an invalid structured job record"
            ) from error


def _fact(payload: dict[str, Any], field_name: str) -> EvidenceBackedJobFact:
    item = _object(payload[field_name], field_name)
    _require_exact_fields(
        item,
        {"value", "citations"},
        field_name=field_name,
    )
    value = item["value"]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name}.value must be a string or null")
    citations = _array(item, "citations")
    return EvidenceBackedJobFact(
        value,
        tuple(_citation(citation) for citation in citations),
    )


def _facts(
    payload: dict[str, Any],
    field_name: str,
) -> tuple[EvidenceBackedJobFact, ...]:
    return tuple(
        _fact({"item": item}, "item")
        for item in _array(payload, field_name)
    )


def _citation(payload: object) -> JobFieldCitation:
    item = _object(payload, "citation")
    _require_exact_fields(
        item,
        {"evidence_id", "quoted_text"},
        field_name="citation",
    )
    evidence_id = item["evidence_id"]
    quoted_text = item["quoted_text"]
    if not isinstance(evidence_id, str) or not isinstance(quoted_text, str):
        raise TypeError("citation fields must be strings")
    return JobFieldCitation(evidence_id, quoted_text)


def _object(payload: object, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{field_name} must be an object")
    return payload


def _array(payload: dict[str, Any], field_name: str) -> list[object]:
    value = payload[field_name]
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _require_exact_fields(
    payload: dict[str, Any],
    expected_fields: set[str],
    *,
    field_name: str,
) -> None:
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        raise TypeError(
            f"{field_name} fields do not match the output contract"
        )
