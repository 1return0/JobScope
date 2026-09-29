from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusedEvidenceRank:
    rank: int
    evidence_id: str
    fusion_score: float
    ranks_by_retriever: tuple[tuple[str, int], ...]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> tuple[FusedEvidenceRank, ...]:
    if len(rankings) < 2:
        raise ValueError("RRF requires at least two retrievers")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")

    scores: dict[str, float] = {}
    ranks_by_evidence: dict[str, dict[str, int]] = {}
    for retriever_name, evidence_ids in rankings.items():
        normalized_name = retriever_name.strip()
        if not normalized_name:
            raise ValueError("retriever name must not be blank")
        normalized_ids = tuple(
            evidence_id.strip() for evidence_id in evidence_ids
        )
        if any(not evidence_id for evidence_id in normalized_ids):
            raise ValueError("evidence ID must not be blank")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError(
                f"{normalized_name} ranking contains duplicate IDs"
            )

        for rank, evidence_id in enumerate(normalized_ids, start=1):
            scores[evidence_id] = scores.get(evidence_id, 0.0) + (
                1 / (rank_constant + rank)
            )
            ranks_by_evidence.setdefault(evidence_id, {})[
                normalized_name
            ] = rank

    ordered = sorted(
        scores,
        key=lambda evidence_id: (-scores[evidence_id], evidence_id),
    )[:top_k]
    return tuple(
        FusedEvidenceRank(
            rank=rank,
            evidence_id=evidence_id,
            fusion_score=scores[evidence_id],
            ranks_by_retriever=tuple(
                sorted(ranks_by_evidence[evidence_id].items())
            ),
        )
        for rank, evidence_id in enumerate(ordered, start=1)
    )
