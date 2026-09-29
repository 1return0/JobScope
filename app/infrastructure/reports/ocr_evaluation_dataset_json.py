from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.application.ocr.ocr_evaluation_dataset import (
    OcrEvaluationDataset,
    OcrGroundTruthCase,
    OcrGroundTruthField,
)


class OcrEvaluationDatasetJsonError(ValueError):
    pass


class OcrEvaluationDatasetFileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedOcrEvaluationCaseFiles:
    case: OcrGroundTruthCase
    artifact_path: Path
    expected_text_path: Path


def load_ocr_evaluation_dataset(
    path: Path,
) -> OcrEvaluationDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _decode_dataset(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise OcrEvaluationDatasetJsonError(
            "OCR evaluation dataset does not match the JSON schema"
        ) from error


def verify_ocr_evaluation_dataset_files(
    dataset: OcrEvaluationDataset,
    *,
    artifact_root: Path,
) -> tuple[VerifiedOcrEvaluationCaseFiles, ...]:
    resolved_root = artifact_root.resolve()
    return tuple(
        _verify_case_files(case, resolved_root)
        for case in dataset.cases
    )


def _decode_dataset(payload: object) -> OcrEvaluationDataset:
    root = _object(payload, "dataset")
    return OcrEvaluationDataset(
        dataset_id=_string(root, "dataset_id"),
        dataset_version=_integer(root, "dataset_version"),
        review_status=_string(root, "review_status"),  # type: ignore[arg-type]
        cases=tuple(
            _decode_case(case) for case in _array(root, "cases")
        ),
    )


def _decode_case(payload: object) -> OcrGroundTruthCase:
    item = _object(payload, "case")
    tags = _array(item, "tags")
    if not all(isinstance(tag, str) for tag in tags):
        raise TypeError("OCR case tags must be strings")
    return OcrGroundTruthCase(
        case_id=_string(item, "case_id"),
        artifact_reference=_string(item, "artifact_reference"),
        artifact_sha256=_string(item, "artifact_sha256"),
        expected_text_reference=_string(
            item,
            "expected_text_reference",
        ),
        expected_text_sha256=_string(
            item,
            "expected_text_sha256",
        ),
        tags=tuple(tags),
        fields=tuple(
            _decode_field(field) for field in _array(item, "fields")
        ),
    )


def _decode_field(payload: object) -> OcrGroundTruthField:
    item = _object(payload, "field")
    critical = item["critical"]
    if not isinstance(critical, bool):
        raise TypeError("OCR field critical must be a boolean")
    return OcrGroundTruthField(
        field_name=_string(item, "field_name"),
        expected_value=_string(item, "expected_value"),
        critical=critical,
    )


def _verify_case_files(
    case: OcrGroundTruthCase,
    resolved_root: Path,
) -> VerifiedOcrEvaluationCaseFiles:
    artifact_path = _resolve_inside(
        resolved_root,
        case.artifact_reference,
    )
    expected_text_path = _resolve_inside(
        resolved_root,
        case.expected_text_reference,
    )
    _verify_hash(artifact_path, case.artifact_sha256)
    _verify_hash(expected_text_path, case.expected_text_sha256)
    return VerifiedOcrEvaluationCaseFiles(
        case=case,
        artifact_path=artifact_path,
        expected_text_path=expected_text_path,
    )


def _resolve_inside(root: Path, reference: str) -> Path:
    relative_path = Path(reference)
    if relative_path.is_absolute():
        raise OcrEvaluationDatasetFileError(
            "OCR dataset file reference must be relative"
        )
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root):
        raise OcrEvaluationDatasetFileError(
            "OCR dataset file reference escapes the artifact root"
        )
    if not resolved.is_file():
        raise OcrEvaluationDatasetFileError(
            f"OCR dataset file does not exist: {reference}"
        )
    return resolved


def _verify_hash(path: Path, expected_sha256: str) -> None:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256.casefold():
        raise OcrEvaluationDatasetFileError(
            f"OCR dataset file hash mismatch: {path.name}"
        )


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"OCR {label} must be an object")
    return value


def _array(payload: dict[str, Any], field: str) -> list[object]:
    value = payload[field]
    if not isinstance(value, list):
        raise TypeError(f"OCR {field} must be an array")
    return value


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise TypeError(f"OCR {field} must be a string")
    return value


def _integer(payload: dict[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"OCR {field} must be an integer")
    return value
