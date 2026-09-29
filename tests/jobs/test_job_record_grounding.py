import unittest

from app.application.jobs.job_record_grounding import (
    JobRecordGroundingValidator,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation
from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    JobFieldCitation,
    StructuredJobRecordDraft,
)


def _chunk() -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id="ev_" + "1" * 64,
        document_sha256="a" * 64,
        source_reference="official-source",
        source_fragment_ordinal=0,
        chunk_ordinal=0,
        chunker_version="structure-v1",
        text="AI Agent实习生，工作地点为上海，专业不限。",
        location=EvidenceLocation(page_number=1),
    )


def _fact(value: str | None, evidence_id: str = ""):
    if value is None:
        return EvidenceBackedJobFact(None)
    return EvidenceBackedJobFact(
        value,
        (JobFieldCitation(evidence_id, value),),
    )


def _record(location_fact: EvidenceBackedJobFact) -> StructuredJobRecordDraft:
    chunk = _chunk()
    return StructuredJobRecordDraft(
        source_snapshot_id="snap_" + "2" * 64,
        extraction_contract_version="job-record-v1",
        company=_fact(None),
        job_title=_fact("AI Agent实习生", chunk.evidence_id),
        locations=(location_fact,),
        education_requirement=_fact(None),
        major_requirement=_fact(None),
        recruitment_type=_fact(None),
        application_deadline=_fact(None),
        responsibilities=(),
        required_qualifications=(),
        preferred_qualifications=(),
    )


class JobRecordGroundingValidatorTest(unittest.TestCase):
    def test_accepts_citations_from_allowed_evidence(self) -> None:
        chunk = _chunk()
        report = JobRecordGroundingValidator().validate(
            _record(_fact("上海", chunk.evidence_id)),
            allowed_chunks=(chunk,),
        )

        self.assertTrue(report.valid)
        self.assertEqual(7, report.checked_fact_count)
        self.assertEqual(2, report.checked_citation_count)

    def test_rejects_citation_outside_allowed_evidence(self) -> None:
        report = JobRecordGroundingValidator().validate(
            _record(_fact("上海", "ev_" + "9" * 64)),
            allowed_chunks=(_chunk(),),
        )

        self.assertFalse(report.valid)
        self.assertEqual(
            "citation-outside-extraction-evidence",
            report.issues[0].code,
        )
        self.assertEqual("locations[0]", report.issues[0].field_path)

    def test_rejects_fabricated_quote(self) -> None:
        chunk = _chunk()
        report = JobRecordGroundingValidator().validate(
            _record(_fact("北京", chunk.evidence_id)),
            allowed_chunks=(chunk,),
        )

        self.assertFalse(report.valid)
        self.assertEqual(
            "quoted-text-not-found-in-evidence",
            report.issues[0].code,
        )


if __name__ == "__main__":
    unittest.main()
