from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.application.retrieval.retrieval_evaluation import (
    GoldenRetrievalCase,
    RelevanceJudgment,
)
from app.application.retrieval.retrieval_experiment import (
    RetrievalEvaluationDataset,
)


class RetrievalEvaluationJsonError(ValueError):
    pass


def load_retrieval_evaluation_dataset(
    path: Path,
) -> RetrievalEvaluationDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("root must be an object")
        cases_payload = payload["cases"]
        if not isinstance(cases_payload, list):
            raise TypeError("cases must be an array")
        cases = tuple(
            _load_case(case_payload)
            for case_payload in cases_payload
        )
        return RetrievalEvaluationDataset(
            dataset_id=_required_string(payload, "dataset_id"),
            annotation_status=_required_string(
                payload,
                "annotation_status",
            ),
            annotated_by=_required_string(payload, "annotated_by"),
            source_reference=_required_string(
                payload,
                "source_reference",
            ),
            document_sha256=_required_string(
                payload,
                "document_sha256",
            ),
            chunker_version=_required_string(
                payload,
                "chunker_version",
            ),
            tokenizer_version=_required_string(
                payload,
                "tokenizer_version",
            ),
            cases=cases,
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise RetrievalEvaluationJsonError(
            f"invalid retrieval evaluation dataset: {error}"
        ) from error


def _load_case(payload: Any) -> GoldenRetrievalCase:
    if not isinstance(payload, dict):
        raise TypeError("each case must be an object")
    judgments_payload = payload["judgments"]
    if not isinstance(judgments_payload, list):
        raise TypeError("judgments must be an array")
    judgments = tuple(
        _load_judgment(judgment_payload)
        for judgment_payload in judgments_payload
    )
    return GoldenRetrievalCase(
        case_id=_required_string(payload, "case_id"),
        query=_required_string(payload, "query"),
        judgments=judgments,
    )


def _load_judgment(payload: Any) -> RelevanceJudgment:
    if not isinstance(payload, dict):
        raise TypeError("each judgment must be an object")
    grade = payload["relevance_grade"]
    if not isinstance(grade, int) or isinstance(grade, bool):
        raise TypeError("relevance_grade must be an integer")
    return RelevanceJudgment(
        evidence_id=_required_string(payload, "evidence_id"),
        relevance_grade=grade,
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value
