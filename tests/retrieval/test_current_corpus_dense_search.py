import unittest

from app.application.retrieval.current_corpus_dense_search import (
    CurrentCorpusDenseSearchService,
)
from app.application.retrieval.dense_retrieval import EmbeddingSpec
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _FakeCorpusReader:
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


class _FakeEmbedder:
    def __init__(self) -> None:
        self._spec = EmbeddingSpec(
            provider="test",
            model_name="fake-dense-model",
            model_revision="test-v1",
            dimension=2,
        )
        self._vectors = {
            "大学不在列表中": (1.0, 0.0),
            "手动输入院校名称": (0.9, 0.1),
            "面试结果通知": (0.0, 1.0),
        }

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vectors[text] for text in texts)

    def embed_query(self, query: str) -> tuple[float, ...]:
        return self._vectors[query]


class CurrentCorpusDenseSearchServiceTest(unittest.TestCase):
    def test_loads_current_chunks_builds_index_and_returns_top_k(
        self,
    ) -> None:
        relevant = self._chunk(1, "手动输入院校名称")
        unrelated = self._chunk(2, "面试结果通知")
        reader = _FakeCorpusReader([relevant, unrelated])
        service = CurrentCorpusDenseSearchService(
            reader,
            _FakeEmbedder(),
            chunker_version="structure-v1",
        )

        report = service.search(
            "大学不在列表中",
            top_k=1,
            source_reference="official-source",
        )

        self.assertEqual(
            [("structure-v1", "official-source")],
            reader.calls,
        )
        self.assertEqual(2, report.corpus_size)
        self.assertEqual(1, len(report.results))
        self.assertEqual(
            relevant.evidence_id,
            report.results[0].chunk.evidence_id,
        )
        self.assertTrue(report.corpus_fingerprint.startswith("corpus_"))
        self.assertTrue(report.embedding_identity.startswith("emb_"))

    def test_rejects_empty_current_corpus(self) -> None:
        service = CurrentCorpusDenseSearchService(
            _FakeCorpusReader([]),
            _FakeEmbedder(),
            chunker_version="structure-v1",
        )

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            service.search("大学不在列表中")

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
