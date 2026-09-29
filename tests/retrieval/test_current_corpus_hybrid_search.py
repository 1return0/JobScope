import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from app.application.retrieval.current_corpus_hybrid_search import (
    CurrentCorpusHybridSearchService,
    HybridSearchCapacityError,
)
from app.application.retrieval.dense_retrieval import EmbeddingSpec
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _Reader:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks
        self.call_count = 0

    def list_current_chunks(
        self,
        *,
        chunker_version: str,
        source_reference: str | None = None,
    ) -> list[EvidenceChunk]:
        self.call_count += 1
        return list(self.chunks)


class _Embedder:
    def __init__(self) -> None:
        self._spec = EmbeddingSpec(
            provider="test",
            model_name="fake",
            model_revision="v1",
            dimension=2,
        )
        self.document_calls = 0
        self.query_calls = 0

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        self.document_calls += 1
        return tuple(
            (1.0, 0.0) if "院校" in text else (0.0, 1.0)
            for text in texts
        )

    def embed_query(self, query: str) -> tuple[float, ...]:
        self.query_calls += 1
        return (1.0, 0.0)


class _BlockingEmbedder(_Embedder):
    def __init__(self) -> None:
        super().__init__()
        self.query_started = Event()
        self.release_query = Event()

    def embed_query(self, query: str) -> tuple[float, ...]:
        self.query_started.set()
        self.release_query.wait(timeout=5)
        return super().embed_query(query)


class CurrentCorpusHybridSearchServiceTest(unittest.TestCase):
    def test_prepare_once_then_reuses_indexes_for_queries(self) -> None:
        reader = _Reader(
            [
                self._chunk(1, "院校信息支持手动输入"),
                self._chunk(2, "面试通知"),
            ]
        )
        embedder = _Embedder()
        service = CurrentCorpusHybridSearchService(
            reader,
            embedder,
            chunker_version="structure-v1",
            candidate_k=2,
            rank_constant=60,
        )

        service.prepare(source_reference="official-source")
        first = service.search("学校名单中没有大学", top_k=1)
        second = service.search("无法选择院校", top_k=1)

        self.assertEqual(1, reader.call_count)
        self.assertEqual(1, embedder.document_calls)
        self.assertEqual(2, embedder.query_calls)
        self.assertEqual(1, len(first.results))
        self.assertEqual(1, len(second.results))

    def test_requires_prepare_before_search(self) -> None:
        service = CurrentCorpusHybridSearchService(
            _Reader([self._chunk(1, "院校信息")]),
            _Embedder(),
            chunker_version="structure-v1",
            candidate_k=1,
            rank_constant=60,
        )

        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            service.search("学校名单中没有大学", top_k=1)

    def test_failed_refresh_keeps_previous_complete_index(self) -> None:
        reader = _Reader([self._chunk(1, "院校信息支持手动输入")])
        service = CurrentCorpusHybridSearchService(
            reader,
            _Embedder(),
            chunker_version="structure-v1",
            candidate_k=1,
            rank_constant=60,
        )
        service.prepare()
        original = service.search("学校名单中没有大学", top_k=1)
        reader.chunks = []

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            service.prepare()

        after_failure = service.search("学校名单中没有大学", top_k=1)
        self.assertEqual(
            original.corpus_fingerprint,
            after_failure.corpus_fingerprint,
        )

    def test_rejects_excess_concurrent_query_without_queueing(self) -> None:
        embedder = _BlockingEmbedder()
        service = CurrentCorpusHybridSearchService(
            _Reader([self._chunk(1, "院校信息支持手动输入")]),
            embedder,
            chunker_version="structure-v1",
            candidate_k=1,
            rank_constant=60,
            max_concurrent_queries=1,
        )
        service.prepare()

        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                service.search,
                "学校名单中没有大学",
                top_k=1,
            )
            self.assertTrue(embedder.query_started.wait(timeout=2))
            with self.assertRaises(HybridSearchCapacityError):
                service.search("无法选择院校", top_k=1)
            embedder.release_query.set()
            first.result(timeout=2)

    def test_limits_results_to_selected_source_without_rebuilding_index(self) -> None:
        reader = _Reader(
            [
                self._chunk(1, "same query text", "source-a"),
                self._chunk(2, "same query text", "source-b"),
            ]
        )
        embedder = _Embedder()
        service = CurrentCorpusHybridSearchService(
            reader,
            embedder,
            chunker_version="structure-v1",
            candidate_k=2,
            rank_constant=60,
        )
        service.prepare()

        report = service.search(
            "same query",
            top_k=2,
            source_references=("source-b",),
        )

        self.assertEqual(1, report.corpus_size)
        self.assertEqual(1, len(report.results))
        self.assertEqual("source-b", report.results[0].chunk.source_reference)
        self.assertEqual(1, embedder.document_calls)

    @staticmethod
    def _chunk(
        index: int,
        text: str,
        source_reference: str = "official-source",
    ) -> EvidenceChunk:
        return EvidenceChunk(
            evidence_id="ev_" + f"{index:064x}",
            document_sha256="a" * 64,
            source_reference=source_reference,
            source_fragment_ordinal=index,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text=text,
            location=EvidenceLocation(heading_path=("FAQ",)),
        )


if __name__ == "__main__":
    unittest.main()
