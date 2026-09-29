import unittest

from app.application.retrieval.reciprocal_rank_fusion import (
    reciprocal_rank_fusion,
)


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_rewards_evidence_supported_by_both_retrievers(self) -> None:
        results = reciprocal_rank_fusion(
            {
                "bm25": ("ev_shared", "ev_lexical"),
                "dense": ("ev_semantic", "ev_shared"),
            },
            top_k=3,
            rank_constant=60,
        )

        self.assertEqual("ev_shared", results[0].evidence_id)
        self.assertEqual(
            (("bm25", 1), ("dense", 2)),
            results[0].ranks_by_retriever,
        )

    def test_keeps_unique_candidates_from_each_retriever(self) -> None:
        results = reciprocal_rank_fusion(
            {
                "bm25": ("ev_lexical",),
                "dense": ("ev_semantic",),
            },
            top_k=2,
        )

        self.assertEqual(
            {"ev_lexical", "ev_semantic"},
            {result.evidence_id for result in results},
        )

    def test_rejects_duplicate_ids_inside_one_ranking(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            reciprocal_rank_fusion(
                {
                    "bm25": ("ev_1", "ev_1"),
                    "dense": ("ev_2",),
                },
                top_k=2,
            )

    def test_requires_two_retrievers_and_positive_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            reciprocal_rank_fusion(
                {"bm25": ("ev_1",)},
                top_k=1,
            )
        with self.assertRaisesRegex(ValueError, "top_k"):
            reciprocal_rank_fusion(
                {"bm25": (), "dense": ()},
                top_k=0,
            )
        with self.assertRaisesRegex(ValueError, "rank_constant"):
            reciprocal_rank_fusion(
                {"bm25": (), "dense": ()},
                top_k=1,
                rank_constant=0,
            )


if __name__ == "__main__":
    unittest.main()
