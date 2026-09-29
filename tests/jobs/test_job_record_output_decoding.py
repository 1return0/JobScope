import json
import unittest

from app.application.jobs.job_record_output_decoding import (
    JobRecordOutputDecoder,
    JobRecordOutputDecodingError,
)


def _fact(value, citations=None):
    return {"value": value, "citations": citations or []}


class JobRecordOutputDecoderTest(unittest.TestCase):
    def test_decodes_evidence_backed_job_record(self) -> None:
        citation = {
            "evidence_id": "ev-1",
            "quoted_text": "工作地点为上海",
        }
        payload = {
            "company": _fact("示例科技", [citation]),
            "job_title": _fact("AI Agent实习生", [citation]),
            "locations": [_fact("上海", [citation])],
            "education_requirement": _fact(None),
            "major_requirement": _fact("专业不限", [citation]),
            "recruitment_type": _fact("校园招聘", [citation]),
            "application_deadline": _fact(None),
            "responsibilities": [],
            "required_qualifications": [],
            "preferred_qualifications": [],
        }

        record = JobRecordOutputDecoder().decode(
            json.dumps(payload, ensure_ascii=False),
            source_snapshot_id="snap-1",
            extraction_contract_version="job-record-v1",
        )

        self.assertEqual("上海", record.locations[0].value)
        self.assertEqual("snap-1", record.source_snapshot_id)
        self.assertFalse(record.application_deadline.is_stated)

    def test_rejects_claimed_value_without_citation(self) -> None:
        payload = {
            "company": _fact("示例科技"),
            "job_title": _fact(None),
            "locations": [],
            "education_requirement": _fact(None),
            "major_requirement": _fact(None),
            "recruitment_type": _fact(None),
            "application_deadline": _fact(None),
            "responsibilities": [],
            "required_qualifications": [],
            "preferred_qualifications": [],
        }

        with self.assertRaises(JobRecordOutputDecodingError):
            JobRecordOutputDecoder().decode(
                json.dumps(payload, ensure_ascii=False),
                source_snapshot_id="snap-1",
                extraction_contract_version="job-record-v1",
            )

    def test_rejects_unknown_model_output_field(self) -> None:
        payload = {
            "company": _fact(None),
            "job_title": _fact(None),
            "locations": [],
            "education_requirement": _fact(None),
            "major_requirement": _fact(None),
            "recruitment_type": _fact(None),
            "application_deadline": _fact(None),
            "responsibilities": [],
            "required_qualifications": [],
            "preferred_qualifications": [],
            "model_comment": "trust me",
        }

        with self.assertRaises(JobRecordOutputDecodingError):
            JobRecordOutputDecoder().decode(
                json.dumps(payload),
                source_snapshot_id="snap-1",
                extraction_contract_version="job-record-v1",
            )

    def test_does_not_blame_model_for_blank_caller_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_snapshot_id"):
            JobRecordOutputDecoder().decode(
                "{}",
                source_snapshot_id=" ",
                extraction_contract_version="job-record-v1",
            )


if __name__ == "__main__":
    unittest.main()
