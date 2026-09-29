import json
import unittest

from app.application.answering.answer_prompting import GroundedAnswerPrompt
from app.application.answering.grounded_answer_service import (
    GroundedAnswerService,
    UngroundedModelAnswerError,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _RecordingAnswerModel:
    def __init__(self, raw_output: str) -> None:
        self._raw_output = raw_output
        self.prompts: list[GroundedAnswerPrompt] = []

    def generate(self, prompt: GroundedAnswerPrompt) -> str:
        self.prompts.append(prompt)
        return self._raw_output


class GroundedAnswerServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = EvidenceChunk(
            evidence_id="ev_" + "1" * 64,
            document_sha256="a" * 64,
            source_reference="official-source",
            source_fragment_ordinal=0,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text="该岗位专业不限。",
            location=EvidenceLocation(heading_path=("岗位要求",)),
        )

    def test_empty_retrieval_refuses_without_calling_model(self) -> None:
        model = _RecordingAnswerModel("should not be used")
        service = GroundedAnswerService(model)

        result = service.answer("岗位限制专业吗？", retrieved_chunks=())

        self.assertEqual("insufficient_evidence", result.draft.status)
        self.assertFalse(result.model_called)
        self.assertEqual([], model.prompts)

    def test_valid_model_answer_passes_complete_pipeline(self) -> None:
        model = _RecordingAnswerModel(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "该岗位不限制专业。",
                            "citations": [
                                {
                                    "evidence_id": self.chunk.evidence_id,
                                    "quoted_text": "专业不限",
                                }
                            ],
                        }
                    ],
                    "fallback_message": None,
                },
                ensure_ascii=False,
            )
        )
        service = GroundedAnswerService(model)

        result = service.answer(
            "岗位限制专业吗？",
            retrieved_chunks=(self.chunk,),
        )

        self.assertEqual("answered", result.draft.status)
        self.assertTrue(result.validation.valid)
        self.assertTrue(result.model_called)
        self.assertEqual(
            (self.chunk.evidence_id,),
            model.prompts[0].allowed_evidence_ids,
        )

    def test_ungrounded_model_answer_never_becomes_result(self) -> None:
        model = _RecordingAnswerModel(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "该岗位只招计算机专业。",
                            "citations": [
                                {
                                    "evidence_id": "ev_" + "9" * 64,
                                    "quoted_text": "只招计算机专业",
                                }
                            ],
                        }
                    ],
                    "fallback_message": None,
                },
                ensure_ascii=False,
            )
        )
        service = GroundedAnswerService(model)

        with self.assertRaises(UngroundedModelAnswerError) as context:
            service.answer(
                "岗位限制专业吗？",
                retrieved_chunks=(self.chunk,),
            )

        self.assertEqual(
            "citation-outside-retrieved-evidence",
            context.exception.report.issues[0].code,
        )


if __name__ == "__main__":
    unittest.main()
