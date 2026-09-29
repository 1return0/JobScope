from __future__ import annotations

import json

from app.application.answering.answer_output_decoding import (
    AnswerOutputDecodingError,
)
from app.application.answering.answer_prompting import GroundedAnswerPrompt
from app.application.answering.grounded_answer_service import GroundedAnswerService
from app.config import load_settings
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation
from app.infrastructure.llm.openai_compatible_answer_model import (
    OpenAiCompatibleAnswerModelConfig,
    build_openai_compatible_answer_model,
)


class _RecordingAnswerModel:
    def __init__(self, model: object) -> None:
        self._model = model
        self.last_raw_output: str | None = None

    def generate(self, prompt: GroundedAnswerPrompt) -> str:
        raw_output = self._model.generate(prompt)  # type: ignore[attr-defined]
        self.last_raw_output = raw_output
        return raw_output


def main() -> int:
    settings = load_settings(dotenv_override=True)
    if not settings.answer_generation_enabled:
        raise RuntimeError(
            "set JOBSCOPE_ANSWER_GENERATION_ENABLED=true before the smoke test"
        )

    model = _RecordingAnswerModel(build_openai_compatible_answer_model(
        OpenAiCompatibleAnswerModelConfig(
            model_name=settings.answer_model_name,
            base_url=settings.answer_model_base_url,
            api_key_environment_variable=(
                settings.answer_model_api_key_environment_variable
            ),
            timeout_seconds=settings.answer_model_timeout_seconds,
            max_completion_tokens=(
                settings.answer_model_max_completion_tokens
            ),
        )
    ))
    chunk = EvidenceChunk(
        evidence_id="ev_" + "1" * 64,
        document_sha256="a" * 64,
        source_reference="local-grounded-answer-smoke-test",
        source_fragment_ordinal=0,
        chunk_ordinal=0,
        chunker_version="structure-v1",
        text="该岗位专业不限，工作地点为上海。",
        location=EvidenceLocation(heading_path=("岗位要求",)),
    )

    try:
        result = GroundedAnswerService(model).answer(
            "这个岗位限制专业吗？",
            retrieved_chunks=(chunk,),
        )
    except AnswerOutputDecodingError as error:
        print(
            json.dumps(
                {
                    "status": "smoke_test_failed",
                    "failure_code": error.failure_code,
                    "raw_model_output": model.last_raw_output,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "model_name": settings.answer_model_name,
                "status": result.draft.status,
                "claims": [
                    {
                        "text": claim.text,
                        "citations": [
                            {
                                "evidence_id": citation.evidence_id,
                                "quoted_text": citation.quoted_text,
                            }
                            for citation in claim.citations
                        ],
                    }
                    for claim in result.draft.claims
                ],
                "grounding_valid": result.validation.valid,
                "model_called": result.model_called,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
