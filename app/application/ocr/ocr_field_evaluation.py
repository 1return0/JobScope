from __future__ import annotations

from dataclasses import dataclass

from app.application.ocr.ocr_evaluation import (
    OcrTextEvaluation,
    evaluate_ocr_text,
)


@dataclass(frozen=True, slots=True)
class OcrFieldJudgment:
    case_id: str
    field_name: str
    expected_value: str
    actual_value: str
    critical: bool = True

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("OCR field case_id must not be blank")
        if not self.field_name.strip():
            raise ValueError("OCR field_name must not be blank")
        if not self.expected_value.strip():
            raise ValueError("OCR expected field value must not be blank")


@dataclass(frozen=True, slots=True)
class OcrFieldEvaluation:
    judgment: OcrFieldJudgment
    text_evaluation: OcrTextEvaluation


@dataclass(frozen=True, slots=True)
class OcrNamedFieldMetric:
    field_name: str
    field_count: int
    exact_match_rate: float


@dataclass(frozen=True, slots=True)
class OcrFieldEvaluationReport:
    fields: tuple[OcrFieldEvaluation, ...]
    overall_exact_match_rate: float
    critical_exact_match_rate: float | None
    metrics_by_field: tuple[OcrNamedFieldMetric, ...]


def evaluate_ocr_fields(
    judgments: tuple[OcrFieldJudgment, ...],
) -> OcrFieldEvaluationReport:
    if not judgments:
        raise ValueError("OCR field judgments must not be empty")
    identities = tuple(
        (item.case_id.strip(), item.field_name.strip())
        for item in judgments
    )
    if len(set(identities)) != len(identities):
        raise ValueError(
            "OCR field judgment identities must be unique"
        )

    fields = tuple(
        OcrFieldEvaluation(
            judgment=judgment,
            text_evaluation=evaluate_ocr_text(
                judgment.expected_value,
                judgment.actual_value,
            ),
        )
        for judgment in judgments
    )
    critical_fields = tuple(
        field for field in fields if field.judgment.critical
    )
    field_names = sorted(
        {field.judgment.field_name for field in fields}
    )
    return OcrFieldEvaluationReport(
        fields=fields,
        overall_exact_match_rate=_exact_match_rate(fields),
        critical_exact_match_rate=(
            _exact_match_rate(critical_fields)
            if critical_fields
            else None
        ),
        metrics_by_field=tuple(
            _build_named_metric(field_name, fields)
            for field_name in field_names
        ),
    )


def _build_named_metric(
    field_name: str,
    fields: tuple[OcrFieldEvaluation, ...],
) -> OcrNamedFieldMetric:
    selected = tuple(
        field
        for field in fields
        if field.judgment.field_name == field_name
    )
    return OcrNamedFieldMetric(
        field_name=field_name,
        field_count=len(selected),
        exact_match_rate=_exact_match_rate(selected),
    )


def _exact_match_rate(
    fields: tuple[OcrFieldEvaluation, ...],
) -> float:
    return (
        sum(field.text_evaluation.exact_match for field in fields)
        / len(fields)
    )
