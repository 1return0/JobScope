from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.application.answering.grounded_answer_evaluation import (
    GroundedAnswerEvaluationCase,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


ReviewStatus = Literal["provisional", "reviewed"]


class GroundedAnswerEvaluationDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GroundedAnswerEvaluationDataset:
    dataset_id: str
    dataset_version: int
    review_status: ReviewStatus
    cases: tuple[GroundedAnswerEvaluationCase, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise GroundedAnswerEvaluationDatasetError(
                "answer dataset_id must not be blank"
            )
        if self.dataset_version < 1:
            raise GroundedAnswerEvaluationDatasetError(
                "answer dataset_version must be positive"
            )
        if self.review_status not in ("provisional", "reviewed"):
            raise GroundedAnswerEvaluationDatasetError(
                "unsupported answer dataset review_status"
            )
        if not self.cases:
            raise GroundedAnswerEvaluationDatasetError(
                "answer evaluation dataset must contain cases"
            )
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise GroundedAnswerEvaluationDatasetError(
                "answer evaluation dataset case IDs must be unique"
            )


def load_grounded_answer_evaluation_dataset(
    path: Path,
) -> GroundedAnswerEvaluationDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GroundedAnswerEvaluationDatasetError(
            "answer evaluation dataset could not be read"
        ) from error
    return decode_grounded_answer_evaluation_dataset(payload)


def decode_grounded_answer_evaluation_dataset(
    payload: object,
) -> GroundedAnswerEvaluationDataset:
    try:
        if not isinstance(payload, dict):
            raise TypeError("dataset must be an object")
        return GroundedAnswerEvaluationDataset(
            dataset_id=_string(payload, "dataset_id"),
            dataset_version=_integer(payload, "dataset_version"),
            review_status=_string(  # type: ignore[arg-type]
                payload,
                "review_status",
            ),
            cases=tuple(
                _decode_case(case)
                for case in _list(payload, "cases")
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, GroundedAnswerEvaluationDatasetError):
            raise
        raise GroundedAnswerEvaluationDatasetError(
            "answer evaluation dataset does not match the schema"
        ) from error


def _decode_case(payload: object) -> GroundedAnswerEvaluationCase:
    if not isinstance(payload, dict):
        raise TypeError("answer evaluation case must be an object")
    return GroundedAnswerEvaluationCase(
        case_id=_string(payload, "case_id"),
        question=_string(payload, "question"),
        expected_status=_string(  # type: ignore[arg-type]
            payload,
            "expected_status",
        ),
        retrieved_chunks=tuple(
            _decode_chunk(chunk)
            for chunk in _list(payload, "retrieved_chunks")
        ),
    )


def _decode_chunk(payload: object) -> EvidenceChunk:
    if not isinstance(payload, dict):
        raise TypeError("answer evaluation chunk must be an object")
    location = payload.get("location", {})
    if not isinstance(location, dict):
        raise TypeError("chunk location must be an object")
    heading_path = location.get("heading_path", [])
    if not isinstance(heading_path, list) or not all(
        isinstance(item, str) for item in heading_path
    ):
        raise TypeError("heading_path must be an array of strings")
    return EvidenceChunk(
        evidence_id=_string(payload, "evidence_id"),
        document_sha256=_string(payload, "document_sha256"),
        source_reference=_string(payload, "source_reference"),
        source_fragment_ordinal=_integer(
            payload,
            "source_fragment_ordinal",
        ),
        chunk_ordinal=_integer(payload, "chunk_ordinal"),
        chunker_version=_string(payload, "chunker_version"),
        text=_string(payload, "text"),
        location=EvidenceLocation(
            page_number=_optional_integer(location, "page_number"),
            heading_path=tuple(heading_path),
            sheet_name=_optional_string(location, "sheet_name"),
            cell_reference=_optional_string(
                location,
                "cell_reference",
            ),
            slide_number=_optional_integer(location, "slide_number"),
            table_number=_optional_integer(location, "table_number"),
            table_row_number=_optional_integer(
                location,
                "table_row_number",
            ),
        ),
    )


def _string(payload: dict[object, object], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _integer(payload: dict[object, object], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _list(payload: dict[object, object], field: str) -> list[object]:
    value = payload[field]
    if not isinstance(value, list):
        raise TypeError(f"{field} must be an array")
    return value


def _optional_string(
    payload: dict[object, object],
    field: str,
) -> str | None:
    value = payload.get(field)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    return value


def _optional_integer(
    payload: dict[object, object],
    field: str,
) -> int | None:
    value = payload.get(field)
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int)
    ):
        raise TypeError(f"{field} must be an integer or null")
    return value
