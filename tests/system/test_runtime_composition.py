import unittest
from unittest.mock import patch

from app.config import Settings
from app.runtime_composition import compose_hybrid_search_service


class RuntimeCompositionTest(unittest.TestCase):
    def test_hybrid_composition_returns_built_service(self) -> None:
        expected_service = object()
        settings = Settings(
            embedding_device="cpu",
            embedding_batch_size=2,
            hybrid_candidate_k=10,
            hybrid_rank_constant=60,
            hybrid_max_concurrent_queries=1,
        )

        with (
            patch(
                "app.runtime_composition.SentenceTransformerTextEmbedder"
            ),
            patch(
                "app.runtime_composition."
                "build_current_corpus_hybrid_search_service",
                return_value=expected_service,
            ) as build_service,
        ):
            actual_service = compose_hybrid_search_service(settings)

        self.assertIs(expected_service, actual_service)
        build_service.assert_called_once()


if __name__ == "__main__":
    unittest.main()
