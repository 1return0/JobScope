import unittest

from app.application.answering.claim_support_evaluation import (
    ClaimSupportEvaluationInput,
    ClaimSupportJudgment,
    GroundedClaimSupportEvaluator,
)
from app.application.answering.grounded_answering import (
    EvidenceCitation,
    GroundedAnswerDraft,
    GroundedClaim,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _KeywordJudge:
    def __init__(self) -> None:
        self.inputs: list[ClaimSupportEvaluationInput] = []

    def judge(
        self,
        evaluation_input: ClaimSupportEvaluationInput,
    ) -> ClaimSupportJudgment:
        self.inputs.append(evaluation_input)
        evidence = "".join(evaluation_input.evidence_texts)
        if "上海" in evidence:
            label = "supported"
        elif "北京" in evidence:
            label = "unsupported"
        else:
            label = "insufficient_context"
        return ClaimSupportJudgment(
            label=label,  # type: ignore[arg-type]
            rationale="test judgment",
        )


class GroundedClaimSupportEvaluatorTest(unittest.TestCase):
    def test_passes_full_cited_evidence_to_judge(self) -> None:
        judge = _KeywordJudge()
        evaluator = GroundedClaimSupportEvaluator(judge)
        chunk = self._chunk("工作地点位于上海，专业不限。")
        draft = self._draft(
            claim_text="工作地点是上海。",
            evidence_id=chunk.evidence_id,
            quoted_text="工作地点位于上海",
        )

        report = evaluator.evaluate(
            draft,
            retrieved_chunks=(chunk,),
        )

        self.assertEqual(1.0, report.support_rate)
        self.assertFalse(report.review_required)
        self.assertEqual(
            ("工作地点位于上海，专业不限。",),
            judge.inputs[0].evidence_texts,
        )

    def test_marks_unsupported_claim_for_review(self) -> None:
        evaluator = GroundedClaimSupportEvaluator(_KeywordJudge())
        chunk = self._chunk("工作地点位于北京。")
        draft = self._draft(
            claim_text="工作地点是上海。",
            evidence_id=chunk.evidence_id,
            quoted_text="工作地点位于北京",
        )

        report = evaluator.evaluate(
            draft,
            retrieved_chunks=(chunk,),
        )

        self.assertEqual(0.0, report.support_rate)
        self.assertTrue(report.review_required)
        self.assertEqual(
            "unsupported",
            report.claims[0].judgment.label,
        )

    @staticmethod
    def _draft(
        *,
        claim_text: str,
        evidence_id: str,
        quoted_text: str,
    ) -> GroundedAnswerDraft:
        return GroundedAnswerDraft(
            status="answered",
            claims=(
                GroundedClaim(
                    text=claim_text,
                    citations=(
                        EvidenceCitation(
                            evidence_id=evidence_id,
                            quoted_text=quoted_text,
                        ),
                    ),
                ),
            ),
        )

    @staticmethod
    def _chunk(text: str) -> EvidenceChunk:
        return EvidenceChunk(
            evidence_id=f"ev_{'a' * 64}",
            document_sha256="b" * 64,
            source_reference="test-source",
            source_fragment_ordinal=0,
            chunk_ordinal=0,
            chunker_version="test-v1",
            text=text,
            location=EvidenceLocation(),
        )


if __name__ == "__main__":
    unittest.main()
