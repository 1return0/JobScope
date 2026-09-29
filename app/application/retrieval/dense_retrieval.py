from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal, Protocol

from app.domain.documents.document_chunking import EvidenceChunk


EmbeddingVector = tuple[float, ...]
NormalizationMode = Literal["none", "l2"]


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    provider: str
    model_name: str
    model_revision: str
    dimension: int
    max_sequence_length: int = 512
    normalization: NormalizationMode = "l2"
    query_instruction: str = ""
    document_instruction: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "provider",
            "model_name",
            "model_revision",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if self.normalization not in ("none", "l2"):
            raise ValueError("unsupported normalization")

    @property
    def identity(self) -> str:
        payload = json.dumps(
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "dimension": self.dimension,
                "max_sequence_length": self.max_sequence_length,
                "normalization": self.normalization,
                "query_instruction": self.query_instruction,
                "document_instruction": self.document_instruction,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"emb_{digest}"


class TextEmbedder(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[EmbeddingVector, ...]: ...

    def embed_query(self, query: str) -> EmbeddingVector: ...


@dataclass(frozen=True, slots=True)
class DenseIndexRecord:
    chunk: EvidenceChunk
    vector: EmbeddingVector


@dataclass(frozen=True, slots=True)
class DenseIndexArtifact:
    corpus_fingerprint: str
    chunker_version: str
    embedding_identity: str
    dimension: int
    records: tuple[DenseIndexRecord, ...]


@dataclass(frozen=True, slots=True)
class DenseSearchResult:
    rank: int
    score: float
    chunk: EvidenceChunk


class DenseIndexBuilder:
    def __init__(self, embedder: TextEmbedder) -> None:
        self._embedder = embedder

    def build(
        self,
        chunks: list[EvidenceChunk],
    ) -> DenseIndexArtifact:
        if not chunks:
            raise ValueError("dense index requires chunks")
        evidence_ids = [chunk.evidence_id for chunk in chunks]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("dense index contains duplicate evidence IDs")
        chunker_versions = {
            chunk.chunker_version for chunk in chunks
        }
        if len(chunker_versions) != 1:
            raise ValueError("dense index mixes chunker versions")

        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.evidence_id,
        )
        vectors = self._embedder.embed_documents(
            tuple(chunk.text for chunk in ordered_chunks)
        )
        if len(vectors) != len(ordered_chunks):
            raise ValueError(
                "embedder returned an unexpected vector count"
            )
        records = tuple(
            DenseIndexRecord(
                chunk=chunk,
                vector=_validate_vector(
                    vector,
                    dimension=self._embedder.spec.dimension,
                    normalization=(
                        self._embedder.spec.normalization
                    ),
                ),
            )
            for chunk, vector in zip(
                ordered_chunks,
                vectors,
                strict=True,
            )
        )
        return DenseIndexArtifact(
            corpus_fingerprint=_build_corpus_fingerprint(
                ordered_chunks
            ),
            chunker_version=next(iter(chunker_versions)),
            embedding_identity=self._embedder.spec.identity,
            dimension=self._embedder.spec.dimension,
            records=records,
        )


class InMemoryDenseRetriever:
    def __init__(
        self,
        index: DenseIndexArtifact,
        embedder: TextEmbedder,
    ) -> None:
        if index.embedding_identity != embedder.spec.identity:
            raise ValueError(
                "dense index and query embedder identities differ"
            )
        if index.dimension != embedder.spec.dimension:
            raise ValueError(
                "dense index and query embedder dimensions differ"
            )
        self._index = index
        self._embedder = embedder

    @property
    def corpus_size(self) -> int:
        return len(self._index.records)

    @property
    def corpus_fingerprint(self) -> str:
        return self._index.corpus_fingerprint

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[DenseSearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_vector = _validate_vector(
            self._embedder.embed_query(normalized_query),
            dimension=self._index.dimension,
            normalization=self._embedder.spec.normalization,
        )
        scored = [
            (
                _cosine_similarity(query_vector, record.vector),
                record.chunk,
            )
            for record in self._index.records
        ]
        scored.sort(
            key=lambda item: (-item[0], item[1].evidence_id)
        )
        return [
            DenseSearchResult(
                rank=rank,
                score=score,
                chunk=chunk,
            )
            for rank, (score, chunk) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]


def _validate_vector(
    vector: EmbeddingVector,
    *,
    dimension: int,
    normalization: NormalizationMode,
) -> EmbeddingVector:
    normalized = tuple(float(value) for value in vector)
    if len(normalized) != dimension:
        raise ValueError("embedding vector dimension mismatch")
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("embedding vector must contain finite values")
    vector_norm = math.sqrt(
        sum(value * value for value in normalized)
    )
    if math.isclose(vector_norm, 0.0):
        raise ValueError("embedding vector must not be zero")
    if normalization == "l2":
        return tuple(value / vector_norm for value in normalized)
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


def _build_corpus_fingerprint(
    chunks: list[EvidenceChunk],
) -> str:
    payload = json.dumps(
        [
            {
                "evidence_id": chunk.evidence_id,
                "document_sha256": chunk.document_sha256,
                "chunker_version": chunk.chunker_version,
            }
            for chunk in chunks
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"corpus_{digest}"
