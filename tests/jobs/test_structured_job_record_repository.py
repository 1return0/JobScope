import unittest
from dataclasses import replace
from datetime import UTC, date, datetime

from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from app.application.jobs.job_record_extraction_service import (
    JobRecordExtractionResult,
)
from app.application.jobs.job_record_grounding import (
    JobRecordGroundingReport,
)
from app.application.jobs.job_record_normalization import (
    ApplicationDeadlineNormalizer,
)
from app.application.jobs.job_record_semantic_support import (
    JobRecordSemanticSupportReport,
)
from app.application.jobs.current_job_records import CurrentJobRecordFilters
from app.domain.jobs.structured_job_record import (
    EvidenceBackedJobFact,
    JobFieldCitation,
    StructuredJobRecordDraft,
)
from app.infrastructure.persistence.database import Base, create_session_factory
from app.infrastructure.persistence.models import (
    DocumentSnapshotChunkRow,
    DocumentSnapshotRow,
    EvidenceChunkRow,
    StructuredJobRecordCitationRow,
    StructuredJobRecordRow,
)
from app.infrastructure.persistence.structured_job_record_repository import (
    JobRecordPersistenceEvidenceError,
    SqlAlchemyStructuredJobRecordRepository,
    StaleJobRecordActivationError,
    UnacceptedJobRecordError,
)
from app.infrastructure.persistence.structured_job_record_query import (
    SqlAlchemyCurrentJobRecordQuery,
)


class SqlAlchemyStructuredJobRecordRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = create_session_factory(self.engine)
        self.repository = SqlAlchemyStructuredJobRecordRepository(
            self.session_factory
        )
        self.query = SqlAlchemyCurrentJobRecordQuery(self.session_factory)
        self.snapshot_id = "snap_" + "1" * 64
        self.evidence_id = "ev_" + "2" * 64
        self._insert_source_graph(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
            linked=True,
        )

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_persists_accepted_record_and_citation_lineage(self) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        deadline = ApplicationDeadlineNormalizer().normalize(
            result.record.application_deadline
        )

        report = self.repository.persist_accepted(
            result,
            deadline=deadline,
        )

        self.assertTrue(report.created)
        self.assertEqual(3, report.citations_inserted)
        with self.session_factory() as session:
            row = session.get(StructuredJobRecordRow, report.record_id)
            citation_count = session.scalar(
                select(func.count()).select_from(
                    StructuredJobRecordCitationRow
                )
            )
        self.assertEqual("AI Agent实习生", row.job_title)
        self.assertEqual(["上海"], row.locations)
        self.assertEqual(date(2026, 9, 30), row.application_deadline_normalized)
        self.assertEqual("normalized", row.deadline_normalization_status)
        self.assertFalse(row.is_current)
        self.assertEqual(3, citation_count)

    def test_explicit_activation_publishes_record_idempotently(self) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        persisted = self.repository.persist_accepted(
            result,
            deadline=ApplicationDeadlineNormalizer().normalize(
                result.record.application_deadline
            ),
        )
        activated_at = datetime(2026, 8, 19, 12, tzinfo=UTC)

        first = self.repository.activate(
            persisted.record_id,
            activated_at=activated_at,
            activation_reference="automatic-after-accepted-v1",
        )
        repeated = self.repository.activate(
            persisted.record_id,
            activated_at=activated_at,
            activation_reference="automatic-after-accepted-v1",
        )

        self.assertTrue(first.changed)
        self.assertFalse(repeated.changed)
        with self.session_factory() as session:
            row = session.get(
                StructuredJobRecordRow,
                persisted.record_id,
            )
        self.assertTrue(row.is_current)
        self.assertEqual(
            "automatic-after-accepted-v1",
            row.activation_reference,
        )

    def test_current_query_requires_record_and_snapshot_to_be_current(
        self,
    ) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        persisted = self.repository.persist_accepted(
            result,
            deadline=ApplicationDeadlineNormalizer().normalize(
                result.record.application_deadline
            ),
        )

        self.assertEqual([], self.query.list_current())
        self.repository.activate(
            persisted.record_id,
            activated_at=datetime(2026, 8, 19, 12, tzinfo=UTC),
            activation_reference="review-ticket-001",
        )
        current = self.query.list_current(
            CurrentJobRecordFilters(
                location="上海",
                deadline_on_or_after=date(2026, 9, 1),
                deadline_on_or_before=date(2026, 9, 30),
            )
        )
        self.assertEqual([persisted.record_id], [item.record_id for item in current])

        with self.session_factory.begin() as session:
            snapshot = session.get(
                DocumentSnapshotRow,
                self.snapshot_id,
            )
            snapshot.is_current = False
        self.assertEqual([], self.query.list_current())

    def test_current_query_filters_by_job_title_and_recruitment_type(
        self,
    ) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        persisted = self.repository.persist_accepted(
            result,
            deadline=ApplicationDeadlineNormalizer().normalize(
                result.record.application_deadline
            ),
        )
        self.repository.activate(
            persisted.record_id,
            activated_at=datetime(2026, 8, 19, 12, tzinfo=UTC),
            activation_reference="review-ticket-001",
        )
        with self.session_factory.begin() as session:
            row = session.get(StructuredJobRecordRow, persisted.record_id)
            row.job_title = "AI Agent Intern"
            row.recruitment_type = "internship"

        matched = self.query.list_current(
            CurrentJobRecordFilters(
                job_title="Agent",
                recruitment_type="internship",
            )
        )
        unmatched = self.query.list_current(
            CurrentJobRecordFilters(
                job_title="Backend",
                recruitment_type="internship",
            )
        )

        self.assertEqual([persisted.record_id], [item.record_id for item in matched])
        self.assertEqual([], unmatched)

    def test_current_query_matches_city_with_or_without_suffix(self) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        persisted = self.repository.persist_accepted(
            result,
            deadline=ApplicationDeadlineNormalizer().normalize(
                result.record.application_deadline
            ),
        )
        self.repository.activate(
            persisted.record_id,
            activated_at=datetime(2026, 8, 19, 12, tzinfo=UTC),
            activation_reference="review-ticket-city-alias",
        )
        with self.session_factory.begin() as session:
            row = session.get(StructuredJobRecordRow, persisted.record_id)
            row.locations = ["北京市"]

        matched = self.query.list_current(
            CurrentJobRecordFilters(location="北京")
        )

        self.assertEqual(
            [persisted.record_id],
            [item.record_id for item in matched],
        )

    def test_current_query_matches_legacy_chinese_recruitment_alias(self) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        persisted = self.repository.persist_accepted(
            result,
            deadline=ApplicationDeadlineNormalizer().normalize(
                result.record.application_deadline
            ),
        )
        self.repository.activate(
            persisted.record_id,
            activated_at=datetime(2026, 8, 19, 12, tzinfo=UTC),
            activation_reference="review-ticket-legacy-alias",
        )
        with self.session_factory.begin() as session:
            row = session.get(StructuredJobRecordRow, persisted.record_id)
            row.recruitment_type = "实习"

        matched = self.query.list_current(
            CurrentJobRecordFilters(recruitment_type="internship")
        )

        self.assertEqual([persisted.record_id], [item.record_id for item in matched])

    def test_rejects_activation_for_stale_source_snapshot(self) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        persisted = self.repository.persist_accepted(
            result,
            deadline=ApplicationDeadlineNormalizer().normalize(
                result.record.application_deadline
            ),
        )
        with self.session_factory.begin() as session:
            snapshot = session.get(
                DocumentSnapshotRow,
                self.snapshot_id,
            )
            snapshot.is_current = False

        with self.assertRaises(StaleJobRecordActivationError):
            self.repository.activate(
                persisted.record_id,
                activated_at=datetime(2026, 8, 19, 12, tzinfo=UTC),
                activation_reference="review-ticket-001",
            )

    def test_repeated_identical_result_is_idempotent(self) -> None:
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        deadline = ApplicationDeadlineNormalizer().normalize(
            result.record.application_deadline
        )
        first = self.repository.persist_accepted(
            result,
            deadline=deadline,
        )

        repeated = self.repository.persist_accepted(
            result,
            deadline=deadline,
        )

        self.assertEqual(first.record_id, repeated.record_id)
        self.assertFalse(repeated.created)
        self.assertEqual(0, repeated.citations_inserted)

    def test_rejects_citation_not_linked_to_source_snapshot(self) -> None:
        foreign_evidence_id = "ev_" + "9" * 64
        self._insert_evidence_only(foreign_evidence_id)
        result = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=foreign_evidence_id,
        )
        deadline = ApplicationDeadlineNormalizer().normalize(
            result.record.application_deadline
        )

        with self.assertRaises(JobRecordPersistenceEvidenceError):
            self.repository.persist_accepted(
                result,
                deadline=deadline,
            )

        self.assertEqual(0, self._record_count())

    def test_rejects_result_that_did_not_pass_validation(self) -> None:
        accepted = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        rejected = replace(accepted, status="semantic_validation_failed")
        deadline = ApplicationDeadlineNormalizer().normalize(
            rejected.record.application_deadline
        )

        with self.assertRaises(UnacceptedJobRecordError):
            self.repository.persist_accepted(
                rejected,
                deadline=deadline,
            )

        self.assertEqual(0, self._record_count())

    def test_rejects_accepted_result_without_job_title(self) -> None:
        accepted = self._accepted_result(
            snapshot_id=self.snapshot_id,
            evidence_id=self.evidence_id,
        )
        empty_record = replace(
            accepted.record,
            job_title=EvidenceBackedJobFact(None),
        )
        empty_result = replace(accepted, record=empty_record)
        deadline = ApplicationDeadlineNormalizer().normalize(
            empty_result.record.application_deadline
        )

        with self.assertRaises(UnacceptedJobRecordError):
            self.repository.persist_accepted(
                empty_result,
                deadline=deadline,
            )

        self.assertEqual(0, self._record_count())

    def _insert_source_graph(
        self,
        *,
        snapshot_id: str,
        evidence_id: str,
        linked: bool,
    ) -> None:
        observed_at = datetime(2026, 8, 19, tzinfo=UTC)
        with self.session_factory.begin() as session:
            session.add(
                DocumentSnapshotRow(
                    snapshot_id=snapshot_id,
                    source_reference=f"source-{snapshot_id}",
                    document_format="pdf",
                    content_sha256="a" * 64,
                    byte_size=100,
                    version_number=1,
                    first_observed_at=observed_at,
                    last_observed_at=observed_at,
                    is_current=True,
                )
            )
            session.add(self._evidence_row(evidence_id))
            session.flush()
            if linked:
                session.add(
                    DocumentSnapshotChunkRow(
                        snapshot_id=snapshot_id,
                        evidence_id=evidence_id,
                    )
                )

    def _insert_evidence_only(self, evidence_id: str) -> None:
        with self.session_factory.begin() as session:
            session.add(self._evidence_row(evidence_id))

    @staticmethod
    def _evidence_row(evidence_id: str) -> EvidenceChunkRow:
        return EvidenceChunkRow(
            evidence_id=evidence_id,
            document_sha256="a" * 64,
            source_reference="official-source",
            source_fragment_ordinal=0,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text=(
                "AI Agent实习生，工作地点为上海，"
                "申请截止日期为2026年9月30日。"
            ),
            heading_path=[],
        )

    @staticmethod
    def _accepted_result(
        *,
        snapshot_id: str,
        evidence_id: str,
    ) -> JobRecordExtractionResult:
        def fact(value: str) -> EvidenceBackedJobFact:
            return EvidenceBackedJobFact(
                value,
                (JobFieldCitation(evidence_id, value),),
            )

        unknown = EvidenceBackedJobFact(None)
        record = StructuredJobRecordDraft(
            source_snapshot_id=snapshot_id,
            extraction_contract_version="job-record-v1",
            company=unknown,
            job_title=fact("AI Agent实习生"),
            locations=(fact("上海"),),
            education_requirement=unknown,
            major_requirement=unknown,
            recruitment_type=unknown,
            application_deadline=fact("2026年9月30日"),
            responsibilities=(),
            required_qualifications=(),
            preferred_qualifications=(),
        )
        return JobRecordExtractionResult(
            status="accepted",
            record=record,
            grounding_report=JobRecordGroundingReport(
                valid=True,
                checked_fact_count=10,
                checked_citation_count=3,
                issues=(),
            ),
            semantic_support_report=JobRecordSemanticSupportReport(
                valid=True,
                supported_count=3,
                unsupported_count=0,
                uncertain_count=0,
                evaluations=(),
                judge_identity="judge-v1",
            ),
            prompt_version="job-record-prompt-v1",
            extraction_contract_version="job-record-v1",
            model_identity="model-v1",
        )

    def _record_count(self) -> int:
        with self.session_factory() as session:
            return session.scalar(
                select(func.count()).select_from(
                    StructuredJobRecordRow
                )
            )


if __name__ == "__main__":
    unittest.main()
