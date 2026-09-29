import unittest

from app.application.documents.corpus_ingestion import (
    CorpusIngestionItemReport,
    CorpusIngestionReport,
    CorpusPersistenceSummary,
)
from scripts.corpus.ingest_corpus_manifest import (
    report_exit_code,
    report_to_payload,
)


class IngestCorpusManifestScriptTest(unittest.TestCase):
    def test_serializes_success_and_failure_for_operators(self) -> None:
        report = CorpusIngestionReport(
            manifest_identity="manifest-123",
            failure_policy="continue",
            items=(
                CorpusIngestionItemReport(
                    source_id="source-1",
                    status="succeeded",
                    chunk_count=2,
                    persistence=CorpusPersistenceSummary(
                        snapshot_created=True,
                        chunks_inserted=2,
                        chunks_skipped=0,
                        links_inserted=2,
                        links_skipped=0,
                    ),
                ),
                CorpusIngestionItemReport(
                    source_id="source-2",
                    status="parsing_failed",
                    failure_code="no-extractable-content",
                ),
            ),
        )

        payload = report_to_payload(report)

        self.assertEqual(2, payload["total"])
        self.assertEqual(1, payload["succeeded"])
        self.assertEqual(1, payload["parsing_failed"])
        self.assertEqual(
            2,
            payload["items"][0]["persistence"]["chunks_inserted"],
        )
        self.assertIsNone(payload["items"][1]["persistence"])
        self.assertEqual(2, report_exit_code(report))

    def test_returns_zero_when_every_document_succeeds(self) -> None:
        report = CorpusIngestionReport(
            manifest_identity="manifest-123",
            failure_policy="stop",
            items=(
                CorpusIngestionItemReport(
                    source_id="source-1",
                    status="succeeded",
                    chunk_count=1,
                    persistence=CorpusPersistenceSummary(
                        snapshot_created=False,
                        chunks_inserted=0,
                        chunks_skipped=1,
                        links_inserted=0,
                        links_skipped=1,
                    ),
                ),
            ),
        )

        self.assertEqual(0, report_exit_code(report))


if __name__ == "__main__":
    unittest.main()
