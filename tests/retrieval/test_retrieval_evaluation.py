import unittest

from app.application.retrieval.lexical_retrieval import Bm25Retriever
from app.application.retrieval.retrieval_evaluation import (
    GoldenRetrievalCase,
    RelevanceJudgment,
    RetrievalEvaluator,
)
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation


class RetrievalEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = RetrievalEvaluator()

    def test_perfect_ranking_has_full_metrics(self) -> None:
        report = self.evaluator.evaluate(
            [
                self._case(
                    "case-1",
                    ("ev-a", 3),
                    ("ev-b", 1),
                )
            ],
            retrieve=lambda query, top_k: [
                "ev-a",
                "ev-b",
                "ev-x",
            ],
            top_k=3,
        )

        metrics = report.cases[0]
        self.assertEqual(1.0, metrics.recall_at_k)
        self.assertEqual(1.0, metrics.reciprocal_rank)
        self.assertEqual(1.0, metrics.ndcg_at_k)
        self.assertEqual(1, metrics.first_relevant_rank)

    def test_partial_ranking_distinguishes_three_metrics(
        self,
    ) -> None:
        report = self.evaluator.evaluate(
            [
                self._case(
                    "case-1",
                    ("ev-a", 3),
                    ("ev-b", 1),
                )
            ],
            retrieve=lambda query, top_k: ["ev-x", "ev-b"],
            top_k=2,
        )

        metrics = report.cases[0]
        self.assertEqual(0.5, metrics.recall_at_k)
        self.assertEqual(0.5, metrics.reciprocal_rank)
        self.assertGreater(metrics.ndcg_at_k, 0.0)
        self.assertLess(metrics.ndcg_at_k, 1.0)
        self.assertEqual(2, metrics.first_relevant_rank)

    def test_report_uses_macro_average_across_cases(self) -> None:
        cases = [
            self._case("case-1", ("ev-a", 1)),
            self._case("case-2", ("ev-b", 1)),
        ]

        report = self.evaluator.evaluate(
            cases,
            retrieve=lambda query, top_k: (
                ["ev-a"] if query == "query-case-1" else []
            ),
            top_k=1,
        )

        self.assertEqual(2, report.case_count)
        self.assertEqual(0.5, report.macro_recall_at_k)
        self.assertEqual(0.5, report.mean_reciprocal_rank)
        self.assertEqual(0.5, report.macro_ndcg_at_k)

    def test_rejects_duplicate_ids_in_retrieval_ranking(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.evaluator.evaluate(
                [self._case("case-1", ("ev-a", 1))],
                retrieve=lambda query, top_k: [
                    "ev-a",
                    "ev-a",
                ],
                top_k=2,
            )

    def test_rejects_invalid_golden_cases_and_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "relevant evidence"):
            GoldenRetrievalCase(
                case_id="case-empty",
                query="query",
                judgments=(),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            self.evaluator.evaluate(
                [self._case("case-1", ("ev-a", 1))],
                retrieve=lambda query, top_k: [],
                top_k=0,
            )

    def test_evaluates_real_bm25_ranking_through_adapter(
        self,
    ) -> None:
        retriever = Bm25Retriever(
            [
                self._chunk(1, "Python RAG Agent"),
                self._chunk(2, "Java Spring Backend"),
            ]
        )

        report = self.evaluator.evaluate(
            [self._case("case-1", ("ev_" + "1" * 64, 1))],
            retrieve=lambda query, top_k: [
                result.chunk.evidence_id
                for result in retriever.search(
                    "Python RAG",
                    top_k=top_k,
                )
            ],
            top_k=1,
        )

        self.assertEqual(1.0, report.macro_recall_at_k)
        self.assertEqual(1.0, report.mean_reciprocal_rank)
        self.assertEqual(1.0, report.macro_ndcg_at_k)

    @staticmethod
    def _case(
        case_id: str,
        *judgments: tuple[str, int],
    ) -> GoldenRetrievalCase:
        return GoldenRetrievalCase(
            case_id=case_id,
            query=f"query-{case_id}",
            judgments=tuple(
                RelevanceJudgment(
                    evidence_id=evidence_id,
                    relevance_grade=grade,
                )
                for evidence_id, grade in judgments
            ),
        )

    @staticmethod
    def _chunk(index: int, text: str) -> EvidenceChunk:
        hex_character = str(index)
        return EvidenceChunk(
            evidence_id="ev_" + hex_character * 64,
            document_sha256=hex_character * 64,
            source_reference=f"source-{index}",
            source_fragment_ordinal=0,
            chunk_ordinal=0,
            chunker_version="structure-v1",
            text=text,
            location=EvidenceLocation(page_number=index),
        )


if __name__ == "__main__":
    unittest.main()
