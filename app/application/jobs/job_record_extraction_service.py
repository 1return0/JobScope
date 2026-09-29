from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from app.application.jobs.job_record_grounding import (
    JobRecordGroundingReport,
    JobRecordGroundingValidator,
)
from app.application.jobs.job_record_output_decoding import (
    JobRecordOutputDecoder,
)
from app.application.jobs.job_record_prompting import (
    JobRecordExtractionPrompt,
    JobRecordExtractionPromptBuilder,
)
from app.application.jobs.job_record_semantic_support import (
    JobRecordSemanticSupportEvaluator,
    JobRecordSemanticSupportReport,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.jobs.structured_job_record import StructuredJobRecordDraft


JobRecordExtractionStatus = Literal[
    "accepted",
    "grounding_failed",
    "semantic_validation_failed",
    "manual_review_required",
]


class JobRecordExtractionModel(Protocol):
    @property
    def identity(self) -> str:
        ...

    def generate(self, prompt: JobRecordExtractionPrompt) -> str:
        ...


@dataclass(frozen=True, slots=True)
class JobRecordExtractionResult:
    status: JobRecordExtractionStatus
    record: StructuredJobRecordDraft
    grounding_report: JobRecordGroundingReport
    semantic_support_report: JobRecordSemanticSupportReport | None
    prompt_version: str
    extraction_contract_version: str
    model_identity: str


class JobRecordExtractionService:
    def __init__(
        self,
        model: JobRecordExtractionModel,
        semantic_support_evaluator: JobRecordSemanticSupportEvaluator,
        *,
        prompt_builder: JobRecordExtractionPromptBuilder | None = None,
        output_decoder: JobRecordOutputDecoder | None = None,
        grounding_validator: JobRecordGroundingValidator | None = None,
    ) -> None:
        if not model.identity.strip():
            raise ValueError("job extraction model identity must not be blank")
        self._model = model
        self._semantic_support_evaluator = semantic_support_evaluator
        self._prompt_builder = (
            prompt_builder or JobRecordExtractionPromptBuilder()
        )
        self._output_decoder = output_decoder or JobRecordOutputDecoder()
        self._grounding_validator = (
            grounding_validator or JobRecordGroundingValidator()
        )

    def extract(
        self,
        *,
        source_snapshot_id: str,
        chunks: tuple[EvidenceChunk, ...],
    ) -> JobRecordExtractionResult:
        prompt = self._prompt_builder.build(
            source_snapshot_id=source_snapshot_id,
            chunks=chunks,
        )
        raw_output = self._model.generate(prompt)
        record = self._output_decoder.decode(
            raw_output,
            source_snapshot_id=source_snapshot_id,
            extraction_contract_version=(
                prompt.extraction_contract_version
            ),
        )
        grounding_report = self._grounding_validator.validate(
            record,
            allowed_chunks=chunks,
        )
        if not grounding_report.valid:
            return JobRecordExtractionResult(
                status="grounding_failed",
                record=record,
                grounding_report=grounding_report,
                semantic_support_report=None,
                prompt_version=prompt.prompt_version,
                extraction_contract_version=(
                    prompt.extraction_contract_version
                ),
                model_identity=self._model.identity,
            )

        semantic_report = self._semantic_support_evaluator.evaluate(record)
        if semantic_report.valid:
            status: JobRecordExtractionStatus = "accepted"
        elif semantic_report.unsupported_count > 0:
            status = "semantic_validation_failed"
        else:
            status = "manual_review_required"
        return JobRecordExtractionResult(
            status=status,
            record=record,
            grounding_report=grounding_report,
            semantic_support_report=semantic_report,
            prompt_version=prompt.prompt_version,
            extraction_contract_version=(
                prompt.extraction_contract_version
            ),
            model_identity=self._model.identity,
        )
