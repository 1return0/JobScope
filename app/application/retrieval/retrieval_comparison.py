from __future__ import annotations

from dataclasses import dataclass

from app.application.retrieval.retrieval_evaluation import (
    RetrievalCaseMetrics,
    RetrievalEvaluationReport,
)


@dataclass(frozen=True, slots=True)
class RetrievalCaseComparison:
    case_id: str
    baseline_recall_at_k: float
    candidate_recall_at_k: float
    recall_delta: float
    baseline_reciprocal_rank: float
    candidate_reciprocal_rank: float
    reciprocal_rank_delta: float
    baseline_ndcg_at_k: float
    candidate_ndcg_at_k: float
    ndcg_delta: float


@dataclass(frozen=True, slots=True)
class RetrievalComparisonReport:
    baseline_name: str
    candidate_name: str
    top_k: int
    case_count: int
    macro_recall_delta: float
    mean_reciprocal_rank_delta: float
    macro_ndcg_delta: float
    cases: tuple[RetrievalCaseComparison, ...]


def compare_retrieval_reports(
    baseline_name: str,
    baseline: RetrievalEvaluationReport,
    candidate_name: str,
    candidate: RetrievalEvaluationReport,
) -> RetrievalComparisonReport:
    normalized_baseline_name = baseline_name.strip()
    normalized_candidate_name = candidate_name.strip()
    if not normalized_baseline_name or not normalized_candidate_name:
        raise ValueError("retriever names must not be blank")
    if normalized_baseline_name == normalized_candidate_name:
        raise ValueError("retriever names must differ")
    if baseline.top_k != candidate.top_k:
        raise ValueError("retrieval reports use different top_k values")

    baseline_by_id = _index_cases(baseline.cases)
    candidate_by_id = _index_cases(candidate.cases)
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("retrieval reports contain different case IDs")

    case_comparisons = tuple(
        _compare_case(
            baseline_by_id[case_id],
            candidate_by_id[case_id],
        )
        for case_id in baseline_by_id
    )
    return RetrievalComparisonReport(
        baseline_name=normalized_baseline_name,
        candidate_name=normalized_candidate_name,
        top_k=baseline.top_k,
        case_count=len(case_comparisons),
        macro_recall_delta=(
            candidate.macro_recall_at_k
            - baseline.macro_recall_at_k
        ),
        mean_reciprocal_rank_delta=(
            candidate.mean_reciprocal_rank
            - baseline.mean_reciprocal_rank
        ),
        macro_ndcg_delta=(
            candidate.macro_ndcg_at_k
            - baseline.macro_ndcg_at_k
        ),
        cases=case_comparisons,
    )


def _index_cases(
    cases: tuple[RetrievalCaseMetrics, ...],
) -> dict[str, RetrievalCaseMetrics]:
    indexed = {case.case_id: case for case in cases}
    if len(indexed) != len(cases):
        raise ValueError("retrieval report contains duplicate case IDs")
    return indexed


def _compare_case(
    baseline: RetrievalCaseMetrics,
    candidate: RetrievalCaseMetrics,
) -> RetrievalCaseComparison:
    return RetrievalCaseComparison(
        case_id=baseline.case_id,
        baseline_recall_at_k=baseline.recall_at_k,
        candidate_recall_at_k=candidate.recall_at_k,
        recall_delta=(
            candidate.recall_at_k - baseline.recall_at_k
        ),
        baseline_reciprocal_rank=baseline.reciprocal_rank,
        candidate_reciprocal_rank=candidate.reciprocal_rank,
        reciprocal_rank_delta=(
            candidate.reciprocal_rank - baseline.reciprocal_rank
        ),
        baseline_ndcg_at_k=baseline.ndcg_at_k,
        candidate_ndcg_at_k=candidate.ndcg_at_k,
        ndcg_delta=candidate.ndcg_at_k - baseline.ndcg_at_k,
    )
