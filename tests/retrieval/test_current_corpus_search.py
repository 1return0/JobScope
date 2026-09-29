import unittest

from app.application.retrieval.current_corpus_search import (
    CurrentCorpusSearchService,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _RecordingCorpusReader:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, str | None]] = []

    def list_current_chunks(
        self,
        *,
        chunker_version: str,
        source_reference: str | None = None,
    ) -> list[EvidenceChunk]:
        self.calls.append((chunker_version, source_reference))
        return list(self._chunks)


class CurrentCorpusSearchServiceTest(unittest.TestCase):
    def test_loads_current_version_and_returns_bm25_report(self) -> None:
        reader = _RecordingCorpusReader(
            [
                self._chunk(1, "campus resume interview"),
                self._chunk(2, "campus application process"),
            ]
        )
        service = CurrentCorpusSearchService(
            reader,
            chunker_version="structure-v1",
        )

        report = service.search(
            "resume interview",
            top_k=1,
            source_reference="official-source",
        )

        self.assertEqual(
            [("structure-v1", "official-source")],
            reader.calls,
        )
        self.assertEqual(2, report.corpus_size)
        self.assertEqual("mixed-cjk-bigram-v1", report.tokenizer_version)
        self.assertEqual(1, len(report.results))
        self.assertEqual(
            "campus resume interview",
            report.results[0].chunk.text,
        )

    def test_rejects_invalid_input_before_reading_database(self) -> None:
        reader = _RecordingCorpusReader([])
        service = CurrentCorpusSearchService(
            reader,
            chunker_version="structure-v1",
        )

        with self.assertRaisesRegex(ValueError, "blank"):
            service.search("   ")
        with self.assertRaisesRegex(ValueError, "positive"):
            service.search("resume", top_k=0)

        self.assertEqual([], reader.calls)

    @staticmethod
    def _chunk(index: int, text: str) -> EvidenceChunk:
        return EvidenceChunk(
            evidence_id="ev_" + f"{index:064x}",
            document_sha256="a" * 64,
            source_reference="official-source",
            source_fragment_ordinal=index,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text=text,
            location=EvidenceLocation(heading_path=("FAQ",)),
        )


if __name__ == "__main__":
    unittest.main()
