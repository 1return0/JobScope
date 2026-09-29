from __future__ import annotations

import math
from dataclasses import dataclass

from app.application.retrieval.dense_retrieval import (
    EmbeddingVector,
    TextEmbedder,
)


@dataclass(frozen=True, slots=True)
class EmbeddingSmokeCase:
    query: str
    positive_document: str
    negative_document: str

    def __post_init__(self) -> None:
        for field_name in (
            "query",
            "positive_document",
            "negative_document",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.positive_document == self.negative_document:
            raise ValueError(
                "positive and negative documents must differ"
            )


@dataclass(frozen=True, slots=True)
class EmbeddingSmokeReport:
    provider: str
    model_name: str
    model_revision: str
    embedding_identity: str
    dimension: int
    query: str
    positive_document: str
    negative_document: str
    positive_similarity: float
    negative_similarity: float
    similarity_margin: float
    semantic_order_passed: bool


class EmbeddingSmokeTestService:
    """Runs one isolated semantic sanity check without a corpus store."""

    def __init__(self, embedder: TextEmbedder) -> None:
        self._embedder = embedder

    def run(self, case: EmbeddingSmokeCase) -> EmbeddingSmokeReport:
        documents = self._embedder.embed_documents(
            (
                case.positive_document,
                case.negative_document,
            )
        )
        if len(documents) != 2:
            raise ValueError(
                "embedder must return exactly two document vectors"
            )

        dimension = self._embedder.spec.dimension
        positive_vector = _validate_vector(
            documents[0],
            dimension=dimension,
        )
        negative_vector = _validate_vector(
            documents[1],
            dimension=dimension,
        )
        query_vector = _validate_vector(
            self._embedder.embed_query(case.query),
            dimension=dimension,
        )
        positive_similarity = _cosine_similarity(
            query_vector,
            positive_vector,
        )
        negative_similarity = _cosine_similarity(
            query_vector,
            negative_vector,
        )
        margin = positive_similarity - negative_similarity
        spec = self._embedder.spec
        return EmbeddingSmokeReport(
            provider=spec.provider,
            model_name=spec.model_name,
            model_revision=spec.model_revision,
            embedding_identity=spec.identity,
            dimension=spec.dimension,
            query=case.query,
            positive_document=case.positive_document,
            negative_document=case.negative_document,
            positive_similarity=positive_similarity,
            negative_similarity=negative_similarity,
            similarity_margin=margin,
            semantic_order_passed=margin > 0,
        )


def _validate_vector(
    vector: EmbeddingVector,
    *,
    dimension: int,
) -> EmbeddingVector:
    normalized = tuple(float(value) for value in vector)
    if len(normalized) != dimension:
        raise ValueError("embedding vector dimension mismatch")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("embedding vector must contain finite values")
    if math.isclose(
        sum(value * value for value in normalized),
        0.0,
    ):
        raise ValueError("embedding vector must not be zero")
    return normalized


def _cosine_similarity(
    left: EmbeddingVector,
    right: EmbeddingVector,
) -> float:
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm)
