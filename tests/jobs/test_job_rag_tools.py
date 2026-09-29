import unittest

from app.application.answering.current_corpus_answering import (
    CurrentCorpusAnswerResult,
)
from app.application.answering.grounded_answer_service import GroundedAnswerResult
from app.application.answering.grounded_answering import (
    CitationValidationReport,
    GroundedAnswerDraft,
)
from app.application.jobs.current_job_records import CurrentStructuredJobRecord
from app.application.jobs.job_rag_tools import (
    AnswerRecruitmentQuestionTool,
    SearchJobsWithEvidenceArguments,
    SearchJobsWithEvidenceTool,
)
from app.application.jobs.job_search_tool import SearchCurrentJobsTool
from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchReport,
)


class _Reader:
    def list_current(self, filters=None):
        return [
            CurrentStructuredJobRecord(
                record_id="job_" + "1" * 64,
                source_snapshot_id="snap_" + "2" * 64,
                company="Example",
                job_title="Agent Intern",
                locations=("Shanghai",),
                education_requirement=None,
                major_requirement=None,
                recruitment_type="internship",
                application_deadline_raw=None,
                application_deadline_normalized=None,
                deadline_normalization_status="not_stated",
                source_reference="https://example.com/jobs/agent",
            )
        ]


class _AnswerService:
    def __init__(self) -> None:
        self.source_references = None

    def answer(self, question, *, top_k=5, source_references=None):
        self.source_references = source_references
        return CurrentCorpusAnswerResult(
            retrieval=CurrentCorpusHybridSearchReport(
                query=question,
                top_k=0,
                candidate_k=20,
                rank_constant=60,
                corpus_size=0,
                chunker_version="structure-v1",
                tokenizer_version="test-v1",
                embedding_identity="test-embedding",
                corpus_fingerprint="corpus_test",
                results=(),
            ),
            answer=GroundedAnswerResult(
                draft=GroundedAnswerDraft(
                    status="insufficient_evidence",
                    claims=(),
                    fallback_message="No evidence.",
                ),
                validation=CitationValidationReport(
                    valid=True,
                    checked_claim_count=0,
                    checked_citation_count=0,
                    issues=(),
                ),
                model_called=False,
            ),
        )


class JobRagToolsTest(unittest.TestCase):
    def test_mixed_tool_scopes_rag_to_sources_of_sql_matches(self) -> None:
        answer_service = _AnswerService()
        search_tool = SearchCurrentJobsTool(_Reader())
        tool = SearchJobsWithEvidenceTool(
            search_tool,
            AnswerRecruitmentQuestionTool(answer_service),
        )

        output = tool.execute(
            SearchJobsWithEvidenceArguments(
                location="Shanghai",
                question="Summarize the qualification requirements.",
            )
        )

        expected_scope = ("https://example.com/jobs/agent",)
        self.assertEqual("found", output.status)
        self.assertEqual(expected_scope, output.source_scope)
        self.assertEqual(expected_scope, answer_service.source_references)
        self.assertIsNotNone(output.evidence_answer)


if __name__ == "__main__":
    unittest.main()
