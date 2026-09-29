import unittest

from app.application.retrieval.retrieval_comparison import (
    compare_retrieval_reports,
)
from app.application.retrieval.retrieval_evaluation import (
    RetrievalCaseMetrics,
    RetrievalEvaluationReport,
)


class RetrievalComparisonTest(unittest.TestCase):
    def test_calculates_aggregate_and_per_case_deltas(self) -> None:
        baseline = self._report(
            top_k=5,
            recall=0.5,
            reciprocal_rank=0.25,
            ndcg=0.2,
        )
        candidate = self._report(
            top_k=5,
            recall=1.0,
            reciprocal_rank=1.0,
            ndcg=0.8,
        )

        report = compare_retrieval_reports(
            "bm25",
            baseline,
            "dense",
            candidate,
        )

        self.assertEqual(0.5, report.macro_recall_delta)
        self.assertEqual(0.75, report.mean_reciprocal_rank_delta)
        self.assertAlmostEqual(0.6, report.macro_ndcg_delta)
        self.assertAlmostEqual(0.6, report.cases[0].ndcg_delta)

    def test_rejects_different_top_k_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "different top_k"):
            compare_retrieval_reports(
                "bm25",
                self._report(top_k=3),
                "dense",
                self._report(top_k=5),
            )

    def test_rejects_different_case_sets(self) -> None:
        baseline = self._report(top_k=5, case_id="case-a")
        candidate = self._report(top_k=5, case_id="case-b")

        with self.assertRaisesRegex(ValueError, "different case IDs"):
            compare_retrieval_reports(
                "bm25",
                baseline,
                "dense",
                candidate,
            )

    @staticmethod
    def _report(
        *,
        top_k: int,
        case_id: str = "case-a",
        recall: float = 1.0,
        reciprocal_rank: float = 1.0,
        ndcg: float = 1.0,
    ) -> RetrievalEvaluationReport:
        case = RetrievalCaseMetrics(
            case_id=case_id,
            recall_at_k=recall,
            reciprocal_rank=reciprocal_rank,
            ndcg_at_k=ndcg,
            first_relevant_rank=1,
            retrieved_evidence_ids=("ev_1",),
        )
        return RetrievalEvaluationReport(
            top_k=top_k,
            case_count=1,
            macro_recall_at_k=recall,
            mean_reciprocal_rank=reciprocal_rank,
            macro_ndcg_at_k=ndcg,
            cases=(case,),
        )


if __name__ == "__main__":
    unittest.main()
