from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    evidence_id: str
    relevance_grade: int = 1

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must not be blank")
        if self.relevance_grade < 1:
            raise ValueError("relevance_grade must be positive")


@dataclass(frozen=True, slots=True)
class GoldenRetrievalCase:
    case_id: str
    query: str
    judgments: tuple[RelevanceJudgment, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be blank")
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if not self.judgments:
            raise ValueError(
                "golden case must contain relevant evidence"
            )
        evidence_ids = [
            judgment.evidence_id
            for judgment in self.judgments
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(
                "golden case contains duplicate evidence IDs"
            )


@dataclass(frozen=True, slots=True)
class RetrievalCaseMetrics:
    case_id: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    first_relevant_rank: int | None
    retrieved_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    top_k: int
    case_count: int
    macro_recall_at_k: float
    mean_reciprocal_rank: float
    macro_ndcg_at_k: float
    cases: tuple[RetrievalCaseMetrics, ...]


RankedEvidenceProvider = Callable[
    [str, int],
    Sequence[str],
]


class RetrievalEvaluator:
    def evaluate(
        self,
        cases: list[GoldenRetrievalCase],
        *,
        retrieve: RankedEvidenceProvider,
        top_k: int,
    ) -> RetrievalEvaluationReport:
        if not cases:
            raise ValueError("evaluation cases must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "evaluation contains duplicate case IDs"
            )

        case_metrics = tuple(
            self._evaluate_case(
                case,
                retrieved_evidence_ids=tuple(
                    retrieve(case.query, top_k)
                )[:top_k],
                top_k=top_k,
            )
            for case in cases
        )
        return RetrievalEvaluationReport(
            top_k=top_k,
            case_count=len(case_metrics),
            macro_recall_at_k=fmean(
                item.recall_at_k for item in case_metrics
            ),
            mean_reciprocal_rank=fmean(
                item.reciprocal_rank for item in case_metrics
            ),
            macro_ndcg_at_k=fmean(
                item.ndcg_at_k for item in case_metrics
            ),
            cases=case_metrics,
        )

    def _evaluate_case(
        self,
        case: GoldenRetrievalCase,
        *,
        retrieved_evidence_ids: tuple[str, ...],
        top_k: int,
    ) -> RetrievalCaseMetrics:
        if any(
            not evidence_id.strip()
            for evidence_id in retrieved_evidence_ids
        ):
            raise ValueError(
                "retrieved evidence IDs must not be blank"
            )
        if len(retrieved_evidence_ids) != len(
            set(retrieved_evidence_ids)
        ):
            raise ValueError(
                "retrieval ranking contains duplicate evidence IDs"
            )

        relevance_by_id = {
            judgment.evidence_id: judgment.relevance_grade
            for judgment in case.judgments
        }
        retrieved_relevant_ids = {
            evidence_id
            for evidence_id in retrieved_evidence_ids
            if evidence_id in relevance_by_id
        }
        recall_at_k = (
            len(retrieved_relevant_ids)
            / len(relevance_by_id)
        )

        first_relevant_rank = next(
            (
                rank
                for rank, evidence_id in enumerate(
                    retrieved_evidence_ids,
                    start=1,
                )
                if evidence_id in relevance_by_id
            ),
            None,
        )
        reciprocal_rank = (
            0.0
            if first_relevant_rank is None
            else 1 / first_relevant_rank
        )

        discounted_cumulative_gain = sum(
            self._discounted_gain(
                relevance_by_id.get(evidence_id, 0),
                rank=rank,
            )
            for rank, evidence_id in enumerate(
                retrieved_evidence_ids,
                start=1,
            )
        )
        ideal_grades = sorted(
            relevance_by_id.values(),
            reverse=True,
        )[:top_k]
        ideal_discounted_cumulative_gain = sum(
            self._discounted_gain(grade, rank=rank)
            for rank, grade in enumerate(
                ideal_grades,
                start=1,
            )
        )
        ndcg_at_k = (
            discounted_cumulative_gain
            / ideal_discounted_cumulative_gain
        )

        return RetrievalCaseMetrics(
            case_id=case.case_id,
            recall_at_k=recall_at_k,
            reciprocal_rank=reciprocal_rank,
            ndcg_at_k=ndcg_at_k,
            first_relevant_rank=first_relevant_rank,
            retrieved_evidence_ids=retrieved_evidence_ids,
        )

    @staticmethod
    def _discounted_gain(
        relevance_grade: int,
        *,
        rank: int,
    ) -> float:
        if relevance_grade <= 0:
            return 0.0
        gain = (2**relevance_grade) - 1
        discount = math.log2(rank + 1)
        return gain / discount
