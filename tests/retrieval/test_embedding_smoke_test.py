import unittest

from app.application.retrieval.dense_retrieval import EmbeddingSpec
from app.application.retrieval.embedding_smoke_test import (
    EmbeddingSmokeCase,
    EmbeddingSmokeTestService,
)
from app.infrastructure.retrieval.embedding_model_presets import (
    BGE_SMALL_ZH_V15,
)


class _FakeEmbedder:
    def __init__(
        self,
        vectors: dict[str, tuple[float, ...]],
    ) -> None:
        self._vectors = vectors
        self._spec = EmbeddingSpec(
            provider="test",
            model_name="fake-embedder",
            model_revision="test-v1",
            dimension=2,
        )

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


class EmbeddingSmokeTestServiceTest(unittest.TestCase):
    def test_reports_expected_semantic_order(self) -> None:
        service = EmbeddingSmokeTestService(
            _FakeEmbedder(
                {
                    "campus question": (1.0, 0.0),
                    "relevant evidence": (0.9, 0.1),
                    "unrelated evidence": (0.0, 1.0),
                }
            )
        )

        report = service.run(
            EmbeddingSmokeCase(
                query="campus question",
                positive_document="relevant evidence",
                negative_document="unrelated evidence",
            )
        )

        self.assertTrue(report.semantic_order_passed)
        self.assertGreater(
            report.positive_similarity,
            report.negative_similarity,
        )
        self.assertEqual(2, report.dimension)

    def test_reports_failure_without_hiding_model_behavior(self) -> None:
        service = EmbeddingSmokeTestService(
            _FakeEmbedder(
                {
                    "query": (1.0, 0.0),
                    "positive": (0.0, 1.0),
                    "negative": (1.0, 0.0),
                }
            )
        )

        report = service.run(
            EmbeddingSmokeCase(
                query="query",
                positive_document="positive",
                negative_document="negative",
            )
        )

        self.assertFalse(report.semantic_order_passed)
        self.assertLess(report.similarity_margin, 0)

    def test_rejects_wrong_vector_dimension(self) -> None:
        service = EmbeddingSmokeTestService(
            _FakeEmbedder(
                {
                    "query": (1.0, 0.0),
                    "positive": (1.0,),
                    "negative": (0.0, 1.0),
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            service.run(
                EmbeddingSmokeCase(
                    query="query",
                    positive_document="positive",
                    negative_document="negative",
                )
            )

    def test_bge_query_instruction_is_not_mojibake(self) -> None:
        self.assertEqual(
            "为这个句子生成表示以用于检索相关文章：",
            BGE_SMALL_ZH_V15.query_instruction,
        )


if __name__ == "__main__":
    unittest.main()
