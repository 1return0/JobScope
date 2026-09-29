import json
import unittest

from app.application.jobs.job_record_extraction_service import (
    JobRecordExtractionService,
)
from app.application.jobs.job_record_semantic_support import (
    JobFactSupportDecision,
    JobRecordSemanticSupportEvaluator,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


def _chunk() -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id="ev_" + "1" * 64,
        document_sha256="a" * 64,
        source_reference="official-source",
        source_fragment_ordinal=0,
        chunk_ordinal=0,
        chunker_version="structure-v1",
        text="AI Agent实习生，工作地点为上海。",
        location=EvidenceLocation(page_number=1),
    )


def _payload(evidence_id: str) -> str:
    citation = {
        "evidence_id": evidence_id,
        "quoted_text": "AI Agent实习生",
    }
    unknown = {"value": None, "citations": []}
    payload = {
        "company": unknown,
        "job_title": {
            "value": "AI Agent实习生",
            "citations": [citation],
        },
        "locations": [],
        "education_requirement": unknown,
        "major_requirement": unknown,
        "recruitment_type": unknown,
        "application_deadline": unknown,
        "responsibilities": [],
        "required_qualifications": [],
        "preferred_qualifications": [],
    }
    return json.dumps(payload, ensure_ascii=False)


class _Model:
    identity = "openai-compatible;model=test-model;temperature=0"

    def __init__(self, raw_output: str) -> None:
        self._raw_output = raw_output
        self.calls = 0

    def generate(self, prompt) -> str:
        self.calls += 1
        return self._raw_output


class _Judge:
    identity = "fake-support-judge-v1"

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, *, field_path, fact):
        self.calls += 1
        return JobFactSupportDecision(
            "supported",
            "citation directly supports the extracted value",
        )


class JobRecordExtractionServiceTest(unittest.TestCase):
    def test_accepts_record_after_both_validation_layers(self) -> None:
        chunk = _chunk()
        model = _Model(_payload(chunk.evidence_id))
        judge = _Judge()
        service = JobRecordExtractionService(
            model,
            JobRecordSemanticSupportEvaluator(judge),
        )

        result = service.extract(
            source_snapshot_id="snap_" + "2" * 64,
            chunks=(chunk,),
        )

        self.assertEqual("accepted", result.status)
        self.assertTrue(result.grounding_report.valid)
        self.assertTrue(result.semantic_support_report.valid)
        self.assertEqual(
            judge.identity,
            result.semantic_support_report.judge_identity,
        )
        self.assertEqual(1, model.calls)
        self.assertEqual(1, judge.calls)
        self.assertEqual(model.identity, result.model_identity)

    def test_skips_semantic_judge_after_grounding_failure(self) -> None:
        chunk = _chunk()
        model = _Model(_payload("ev_" + "9" * 64))
        judge = _Judge()
        service = JobRecordExtractionService(
            model,
            JobRecordSemanticSupportEvaluator(judge),
        )

        result = service.extract(
            source_snapshot_id="snap_" + "2" * 64,
            chunks=(chunk,),
        )

        self.assertEqual("grounding_failed", result.status)
        self.assertIsNone(result.semantic_support_report)
        self.assertEqual(0, judge.calls)


if __name__ == "__main__":
    unittest.main()
