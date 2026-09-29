import unittest

from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    JobFieldCitation,
    StructuredJobRecordDraft,
)


def _citation(text: str = "工作地点为上海") -> JobFieldCitation:
    return JobFieldCitation("ev-1", text)


def _fact(value: str) -> EvidenceBackedJobFact:
    return EvidenceBackedJobFact(value, (_citation(value),))


class EvidenceBackedJobFactTest(unittest.TestCase):
    def test_distinguishes_stated_fact_from_unknown_fact(self) -> None:
        stated = EvidenceBackedJobFact(" 上海 ", (_citation(),))
        unknown = EvidenceBackedJobFact(None)

        self.assertEqual("上海", stated.value)
        self.assertTrue(stated.is_stated)
        self.assertFalse(unknown.is_stated)

    def test_rejects_stated_value_without_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one citation"):
            EvidenceBackedJobFact("上海")

    def test_rejects_citation_for_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain"):
            EvidenceBackedJobFact(None, (_citation(),))


class StructuredJobRecordDraftTest(unittest.TestCase):
    def test_preserves_field_level_evidence_and_unknown_values(self) -> None:
        record = StructuredJobRecordDraft(
            source_snapshot_id="snap-1",
            extraction_contract_version="job-record-v1",
            company=_fact("示例科技"),
            job_title=_fact("AI Agent实习生"),
            locations=(_fact("上海"),),
            education_requirement=EvidenceBackedJobFact(None),
            major_requirement=_fact("专业不限"),
            recruitment_type=_fact("校园招聘"),
            application_deadline=EvidenceBackedJobFact(None),
            responsibilities=(_fact("构建智能体工作流"),),
            required_qualifications=(_fact("熟悉Python"),),
            preferred_qualifications=(),
        )

        self.assertEqual("snap-1", record.source_snapshot_id)
        self.assertFalse(record.education_requirement.is_stated)
        self.assertEqual(
            "专业不限",
            record.major_requirement.value,
        )

    def test_rejects_duplicate_list_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "locations values"):
            StructuredJobRecordDraft(
                source_snapshot_id="snap-1",
                extraction_contract_version="job-record-v1",
                company=_fact("示例科技"),
                job_title=_fact("AI Agent实习生"),
                locations=(_fact("上海"), _fact(" 上海 ")),
                education_requirement=EvidenceBackedJobFact(None),
                major_requirement=EvidenceBackedJobFact(None),
                recruitment_type=EvidenceBackedJobFact(None),
                application_deadline=EvidenceBackedJobFact(None),
                responsibilities=(),
                required_qualifications=(),
                preferred_qualifications=(),
            )


if __name__ == "__main__":
    unittest.main()
