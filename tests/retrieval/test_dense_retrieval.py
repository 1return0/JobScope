import unittest

from app.application.retrieval.dense_retrieval import (
    DenseIndexBuilder,
    EmbeddingSpec,
    InMemoryDenseRetriever,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class _FakeEmbedder:
    def __init__(
        self,
        vectors_by_text: dict[str, tuple[float, ...]],
        *,
        revision: str = "test-v1",
    ) -> None:
        self._vectors_by_text = vectors_by_text
        self._spec = EmbeddingSpec(
            provider="test",
            model_name="deterministic-embedder",
            model_revision=revision,
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
        return tuple(self._vectors_by_text[text] for text in texts)

    def embed_query(self, query: str) -> tuple[float, ...]:
        self.query_calls += 1
        return self._vectors_by_text[query]


class DenseRetrievalTest(unittest.TestCase):
    def test_builds_documents_once_and_ranks_by_cosine_similarity(
        self,
    ) -> None:
        embedder = _FakeEmbedder(
            {
                "submit profile": (1.0, 0.0),
                "apply with resume": (0.9, 0.1),
                "interview result": (0.0, 1.0),
            }
        )
        chunks = [
            self._chunk(1, "apply with resume"),
            self._chunk(2, "interview result"),
        ]
        index = DenseIndexBuilder(embedder).build(chunks)
        retriever = InMemoryDenseRetriever(index, embedder)

        results = retriever.search("submit profile", top_k=2)

        self.assertEqual(1, embedder.document_calls)
        self.assertEqual(1, embedder.query_calls)
        self.assertEqual(chunks[0].evidence_id, results[0].chunk.evidence_id)
        self.assertGreater(results[0].score, results[1].score)

    def test_corpus_fingerprint_is_stable_across_input_order(self) -> None:
        embedder = _FakeEmbedder(
            {
                "first": (1.0, 0.0),
                "second": (0.0, 1.0),
            }
        )
        first = self._chunk(1, "first")
        second = self._chunk(2, "second")

        left = DenseIndexBuilder(embedder).build([first, second])
        right = DenseIndexBuilder(embedder).build([second, first])

        self.assertEqual(
            left.corpus_fingerprint,
            right.corpus_fingerprint,
        )

    def test_rejects_query_embedder_with_different_revision(self) -> None:
        vectors = {"document": (1.0, 0.0)}
        builder_embedder = _FakeEmbedder(vectors, revision="v1")
        query_embedder = _FakeEmbedder(vectors, revision="v2")
        index = DenseIndexBuilder(builder_embedder).build(
            [self._chunk(1, "document")]
        )

        with self.assertRaisesRegex(ValueError, "identities differ"):
            InMemoryDenseRetriever(index, query_embedder)

    def test_rejects_wrong_dimension_and_zero_vector(self) -> None:
        wrong_dimension = _FakeEmbedder({"document": (1.0,)})
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            DenseIndexBuilder(wrong_dimension).build(
                [self._chunk(1, "document")]
            )

        zero_vector = _FakeEmbedder({"document": (0.0, 0.0)})
        with self.assertRaisesRegex(ValueError, "must not be zero"):
            DenseIndexBuilder(zero_vector).build(
                [self._chunk(1, "document")]
            )

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
