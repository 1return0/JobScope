import unittest

from app.application.answering.grounded_answering import (
    EvidenceCitation,
    EvidenceGroundingValidator,
    GroundedAnswerDraft,
    GroundedClaim,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class EvidenceGroundingValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = EvidenceChunk(
            evidence_id="ev_" + "1" * 64,
            document_sha256="a" * 64,
            source_reference="official-source",
            source_fragment_ordinal=1,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text="院校列表中没有学校时，可以手动输入标准院校名称。",
            location=EvidenceLocation(heading_path=("院校信息",)),
        )
        self.validator = EvidenceGroundingValidator()

    def test_accepts_quote_from_retrieved_evidence(self) -> None:
        draft = self._answered_draft(
            self.chunk.evidence_id,
            "可以手动输入标准院校名称",
        )

        report = self.validator.validate(
            draft,
            retrieved_chunks=(self.chunk,),
        )

        self.assertTrue(report.valid)
        self.assertEqual(1, report.checked_claim_count)
        self.assertEqual(1, report.checked_citation_count)

    def test_rejects_citation_outside_current_retrieval(self) -> None:
        draft = self._answered_draft(
            "ev_" + "9" * 64,
            "可以手动输入标准院校名称",
        )

        report = self.validator.validate(
            draft,
            retrieved_chunks=(self.chunk,),
        )

        self.assertFalse(report.valid)
        self.assertEqual(
            "citation-outside-retrieved-evidence",
            report.issues[0].code,
        )

    def test_rejects_fabricated_quote(self) -> None:
        draft = self._answered_draft(
            self.chunk.evidence_id,
            "所有学校都必须从列表选择",
        )

        report = self.validator.validate(
            draft,
            retrieved_chunks=(self.chunk,),
        )

        self.assertFalse(report.valid)
        self.assertEqual(
            "quoted-text-not-found-in-evidence",
            report.issues[0].code,
        )

    def test_insufficient_evidence_has_no_factual_claims(self) -> None:
        draft = GroundedAnswerDraft(
            status="insufficient_evidence",
            claims=(),
            fallback_message="当前证据不足，请查看官方招聘页面。",
        )

        report = self.validator.validate(
            draft,
            retrieved_chunks=(self.chunk,),
        )

        self.assertTrue(report.valid)
        self.assertEqual(0, report.checked_citation_count)

    def test_rejects_duplicate_citation_inside_one_claim(self) -> None:
        citation = EvidenceCitation(
            evidence_id=self.chunk.evidence_id,
            quoted_text="手动输入标准院校名称",
        )
        draft = GroundedAnswerDraft(
            status="answered",
            claims=(
                GroundedClaim(
                    text="可以手动输入标准院校名称。",
                    citations=(citation, citation),
                ),
            ),
        )

        report = self.validator.validate(
            draft,
            retrieved_chunks=(self.chunk,),
        )

        self.assertFalse(report.valid)
        self.assertEqual(
            "duplicate-citation-in-claim",
            report.issues[0].code,
        )

    def test_answer_contract_rejects_claimless_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain claims"):
            GroundedAnswerDraft(status="answered", claims=())

    @staticmethod
    def _answered_draft(
        evidence_id: str,
        quoted_text: str,
    ) -> GroundedAnswerDraft:
        return GroundedAnswerDraft(
            status="answered",
            claims=(
                GroundedClaim(
                    text="可以手动输入标准院校名称。",
                    citations=(
                        EvidenceCitation(
                            evidence_id=evidence_id,
                            quoted_text=quoted_text,
                        ),
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
