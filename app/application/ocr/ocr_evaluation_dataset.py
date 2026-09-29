from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal


OcrDatasetReviewStatus = Literal["provisional", "reviewed"]


@dataclass(frozen=True, slots=True)
class OcrGroundTruthField:
    field_name: str
    expected_value: str
    critical: bool = True

    def __post_init__(self) -> None:
        if not self.field_name.strip():
            raise ValueError("OCR ground-truth field name must not be blank")
        if not self.expected_value.strip():
            raise ValueError("OCR expected field value must not be blank")


@dataclass(frozen=True, slots=True)
class OcrGroundTruthCase:
    case_id: str
    artifact_reference: str
    artifact_sha256: str
    expected_text_reference: str
    expected_text_sha256: str
    tags: tuple[str, ...]
    fields: tuple[OcrGroundTruthField, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("OCR ground-truth case_id must not be blank")
        if not self.artifact_reference.strip():
            raise ValueError("OCR artifact reference must not be blank")
        if not self.expected_text_reference.strip():
            raise ValueError(
                "OCR expected text reference must not be blank"
            )
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        _validate_sha256(
            self.expected_text_sha256,
            "expected_text_sha256",
        )
        if not self.tags or any(not tag.strip() for tag in self.tags):
            raise ValueError("OCR case tags must contain non-blank values")
        field_names = tuple(field.field_name for field in self.fields)
        if len(set(field_names)) != len(field_names):
            raise ValueError("OCR field names must be unique within a case")


@dataclass(frozen=True, slots=True)
class OcrEvaluationDataset:
    dataset_id: str
    dataset_version: int
    review_status: OcrDatasetReviewStatus
    cases: tuple[OcrGroundTruthCase, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("OCR dataset_id must not be blank")
        if self.dataset_version < 1:
            raise ValueError("OCR dataset_version must be positive")
        if self.review_status not in ("provisional", "reviewed"):
            raise ValueError("unsupported OCR dataset review_status")
        if not self.cases:
            raise ValueError("OCR evaluation dataset must contain cases")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("OCR evaluation case IDs must be unique")

    @property
    def dataset_sha256(self) -> str:
        canonical_json = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical_json).hexdigest()


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a SHA-256 hex digest"
        ) from error
