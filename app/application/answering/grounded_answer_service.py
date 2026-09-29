from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.application.answering.answer_output_decoding import (
    GroundedAnswerOutputDecoder,
)
from app.application.answering.answer_prompting import (
    GroundedAnswerPrompt,
    GroundedAnswerPromptBuilder,
)
from app.application.answering.grounded_answering import (
    CitationValidationReport,
    EvidenceGroundingValidator,
    GroundedAnswerDraft,
)
from app.domain.documents.document_chunking import EvidenceChunk


class GroundedAnswerModel(Protocol):
    def generate(self, prompt: GroundedAnswerPrompt) -> str:
        ...


class UngroundedModelAnswerError(RuntimeError):
    def __init__(self, report: CitationValidationReport) -> None:
        super().__init__("model answer failed evidence grounding validation")
        self.report = report


@dataclass(frozen=True, slots=True)
class GroundedAnswerResult:
    draft: GroundedAnswerDraft
    validation: CitationValidationReport
    model_called: bool


class GroundedAnswerService:
    def __init__(
        self,
        model: GroundedAnswerModel,
        *,
        prompt_builder: GroundedAnswerPromptBuilder | None = None,
        output_decoder: GroundedAnswerOutputDecoder | None = None,
        grounding_validator: EvidenceGroundingValidator | None = None,
    ) -> None:
        self._model = model
        self._prompt_builder = prompt_builder or GroundedAnswerPromptBuilder()
        self._output_decoder = output_decoder or GroundedAnswerOutputDecoder()
        self._grounding_validator = (
            grounding_validator or EvidenceGroundingValidator()
        )

    def answer(
        self,
        question: str,
        *,
        retrieved_chunks: tuple[EvidenceChunk, ...],
    ) -> GroundedAnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be blank")
        if not retrieved_chunks:
            draft = GroundedAnswerDraft(
                status="insufficient_evidence",
                claims=(),
                fallback_message=(
                    "当前检索结果中没有足够证据，无法可靠回答。"
                ),
            )
            return GroundedAnswerResult(
                draft=draft,
                validation=self._grounding_validator.validate(
                    draft,
                    retrieved_chunks=(),
                ),
                model_called=False,
            )

        prompt = self._prompt_builder.build(
            normalized_question,
            retrieved_chunks=retrieved_chunks,
        )
        raw_output = self._model.generate(prompt)
        draft = self._output_decoder.decode(raw_output)
        validation = self._grounding_validator.validate(
            draft,
            retrieved_chunks=retrieved_chunks,
        )
        if not validation.valid:
            raise UngroundedModelAnswerError(validation)
        return GroundedAnswerResult(
            draft=draft,
            validation=validation,
            model_called=True,
        )
