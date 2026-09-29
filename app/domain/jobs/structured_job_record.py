from __future__ import annotations

from dataclasses import dataclass


def _required(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


@dataclass(frozen=True, slots=True)
class JobFieldCitation:
    evidence_id: str
    quoted_text: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _required(self.evidence_id, field_name="evidence_id"),
        )
        object.__setattr__(
            self,
            "quoted_text",
            _required(self.quoted_text, field_name="quoted_text"),
        )


@dataclass(frozen=True, slots=True)
class EvidenceBackedJobFact:
    value: str | None
    citations: tuple[JobFieldCitation, ...] = ()

    def __post_init__(self) -> None:
        normalized_value = self.value.strip() if self.value else None
        if normalized_value == "":
            normalized_value = None
        object.__setattr__(self, "value", normalized_value)

        if normalized_value is None and self.citations:
            raise ValueError(
                "unknown job fact must not contain citations"
            )
        if normalized_value is not None and not self.citations:
            raise ValueError(
                "stated job fact must contain at least one citation"
            )
        citation_keys = tuple(
            (citation.evidence_id, citation.quoted_text)
            for citation in self.citations
        )
        if len(set(citation_keys)) != len(citation_keys):
            raise ValueError("job fact citations must be unique")

    @property
    def is_stated(self) -> bool:
        return self.value is not None


@dataclass(frozen=True, slots=True)
class StructuredJobRecordDraft:
    source_snapshot_id: str
    extraction_contract_version: str
    company: EvidenceBackedJobFact
    job_title: EvidenceBackedJobFact
    locations: tuple[EvidenceBackedJobFact, ...]
    education_requirement: EvidenceBackedJobFact
    major_requirement: EvidenceBackedJobFact
    recruitment_type: EvidenceBackedJobFact
    application_deadline: EvidenceBackedJobFact
    responsibilities: tuple[EvidenceBackedJobFact, ...]
    required_qualifications: tuple[EvidenceBackedJobFact, ...]
    preferred_qualifications: tuple[EvidenceBackedJobFact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_snapshot_id",
            _required(
                self.source_snapshot_id,
                field_name="source_snapshot_id",
            ),
        )
        object.__setattr__(
            self,
            "extraction_contract_version",
            _required(
                self.extraction_contract_version,
                field_name="extraction_contract_version",
            ),
        )
        for field_name in (
            "locations",
            "responsibilities",
            "required_qualifications",
            "preferred_qualifications",
        ):
            _reject_duplicate_values(
                getattr(self, field_name),
                field_name=field_name,
            )


def _reject_duplicate_values(
    facts: tuple[EvidenceBackedJobFact, ...],
    *,
    field_name: str,
) -> None:
    normalized_values = tuple(
        fact.value.casefold()
        for fact in facts
        if fact.value is not None
    )
    if len(set(normalized_values)) != len(normalized_values):
        raise ValueError(f"{field_name} values must be unique")
