from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel


class EvaluationArtifactSummary(BaseModel):
    artifact_id: str
    artifact_type: Literal["dataset", "experiment"]
    family: str
    filename: str
    relative_path: str
    byte_size: int
    content_sha256: str
    modified_at: datetime
    dataset_id: str | None
    dataset_version: int | None
    review_status: str | None
    case_count: int | None
    executed_at: str | None
    metrics: dict[str, float]


class EvaluationCatalogResponse(BaseModel):
    generated_at: datetime
    artifacts: list[EvaluationArtifactSummary]


class EvaluationArtifactResponse(BaseModel):
    summary: EvaluationArtifactSummary
    payload: dict[str, Any] | list[Any]


def build_evaluation_router(
    data_root: Path | None = None,
) -> APIRouter:
    project_root = Path(__file__).resolve().parents[3]
    resolved_data_root = (data_root or project_root / "data").resolve()
    router = APIRouter(prefix="/v1/evaluations", tags=["evaluations"])

    @router.get(
        "/catalog",
        response_model=EvaluationCatalogResponse,
        summary="List immutable evaluation datasets and experiment reports",
    )
    def list_evaluation_artifacts() -> EvaluationCatalogResponse:
        return EvaluationCatalogResponse(
            generated_at=datetime.now(timezone.utc),
            artifacts=[
                summary
                for _, summary, _ in _load_artifacts(resolved_data_root)
            ],
        )

    @router.get(
        "/artifacts/{artifact_id}",
        response_model=EvaluationArtifactResponse,
        summary="Read one evaluation artifact by its stable catalog identity",
    )
    def get_evaluation_artifact(
        artifact_id: str,
    ) -> EvaluationArtifactResponse:
        for current_id, summary, payload in _load_artifacts(
            resolved_data_root
        ):
            if current_id == artifact_id:
                return EvaluationArtifactResponse(
                    summary=summary,
                    payload=payload,
                )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation artifact not found",
        )

    return router


def _load_artifacts(
    data_root: Path,
) -> list[tuple[str, EvaluationArtifactSummary, dict[str, Any] | list[Any]]]:
    locations = (
        ("dataset", data_root / "evaluation"),
        ("experiment", data_root / "experiments"),
    )
    artifacts = []
    for artifact_type, directory in locations:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                raw = path.read_bytes()
                payload = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, (dict, list)):
                continue
            relative_path = path.relative_to(data_root).as_posix()
            artifact_id = hashlib.sha256(
                relative_path.encode("utf-8")
            ).hexdigest()[:20]
            stat = path.stat()
            metadata = payload if isinstance(payload, dict) else {}
            raw_metrics = metadata.get("metrics")
            metrics = {
                str(name): float(value)
                for name, value in (
                    raw_metrics.items()
                    if isinstance(raw_metrics, dict)
                    else ()
                )
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
            cases = metadata.get("cases")
            case_count = metadata.get("case_count")
            if not isinstance(case_count, int) and isinstance(cases, list):
                case_count = len(cases)
            dataset_version = metadata.get("dataset_version")
            if not isinstance(dataset_version, int):
                dataset_version = None
            summary = EvaluationArtifactSummary(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                family=_family_from_path(relative_path),
                filename=path.name,
                relative_path=relative_path,
                byte_size=stat.st_size,
                content_sha256=hashlib.sha256(raw).hexdigest(),
                modified_at=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ),
                dataset_id=_optional_string(metadata.get("dataset_id")),
                dataset_version=dataset_version,
                review_status=_optional_string(
                    metadata.get("review_status")
                    or metadata.get("annotation_status")
                ),
                case_count=case_count if isinstance(case_count, int) else None,
                executed_at=_optional_string(metadata.get("executed_at")),
                metrics=metrics,
            )
            artifacts.append((artifact_id, summary, payload))
    artifacts.sort(
        key=lambda item: (
            item[1].artifact_type != "experiment",
            item[1].modified_at,
        ),
        reverse=True,
    )
    return artifacts


def _family_from_path(relative_path: str) -> str:
    lowered = relative_path.casefold()
    if "job-agent" in lowered or "planner" in lowered:
        return "Agent Planner"
    if "ocr" in lowered:
        return "OCR"
    if "grounded-answer" in lowered:
        return "Grounded Answer"
    if "retrieval" in lowered:
        return "Retrieval"
    return "Evaluation"


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
