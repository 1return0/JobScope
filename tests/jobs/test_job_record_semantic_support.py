import unittest

from app.application.jobs.job_record_semantic_support import (
    JobFactSupportDecision,
    JobRecordSemanticSupportEvaluator,
)
from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    JobFieldCitation,
    StructuredJobRecordDraft,
)


class _Judge:
    identity = "fake-support-judge-v1"

    def judge(self, *, field_path, fact):
        if field_path == "locations[0]":
            return JobFactSupportDecision(
                "unsupported",
                "citation states Shanghai but value is Beijing",
            )
        return JobFactSupportDecision(
            "supported",
            "citation directly states the value",
        )


def _fact(value: str | None) -> EvidenceBackedJobFact:
    if value is None:
        return EvidenceBackedJobFact(None)
    return EvidenceBackedJobFact(
        value,
        (JobFieldCitation("ev-1", "工作地点为上海"),),
    )


def _record() -> StructuredJobRecordDraft:
    return StructuredJobRecordDraft(
        source_snapshot_id="snap-1",
        extraction_contract_version="job-record-v1",
        company=_fact(None),
        job_title=_fact("AI Agent实习生"),
        locations=(_fact("北京"),),
        education_requirement=_fact(None),
        major_requirement=_fact(None),
        recruitment_type=_fact(None),
        application_deadline=_fact(None),
        responsibilities=(),
        required_qualifications=(),
        preferred_qualifications=(),
    )


class JobRecordSemanticSupportEvaluatorTest(unittest.TestCase):
    def test_evaluates_only_stated_facts_and_rejects_unsupported(self) -> None:
        report = JobRecordSemanticSupportEvaluator(_Judge()).evaluate(
            _record()
        )

        self.assertFalse(report.valid)
        self.assertEqual(1, report.supported_count)
        self.assertEqual(1, report.unsupported_count)
        self.assertEqual(0, report.uncertain_count)
        self.assertEqual(
            ("job_title", "locations[0]"),
            tuple(item.field_path for item in report.evaluations),
        )

    def test_uncertain_is_not_treated_as_valid_support(self) -> None:
        class _UncertainJudge:
            identity = "fake-uncertain-judge-v1"

            def judge(self, *, field_path, fact):
                return JobFactSupportDecision(
                    "uncertain",
                    "citation is ambiguous",
                )

        report = JobRecordSemanticSupportEvaluator(
            _UncertainJudge()
        ).evaluate(_record())

        self.assertFalse(report.valid)
        self.assertEqual(2, report.uncertain_count)


if __name__ == "__main__":
    unittest.main()
