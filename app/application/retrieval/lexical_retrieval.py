from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.domain.documents.document_chunking import EvidenceChunk


_LEXICAL_SEGMENT_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+"
    r"|[A-Za-z0-9][A-Za-z0-9+#._-]*"
)
_CJK_PATTERN = re.compile(
    r"^[\u3400-\u4dbf\u4e00-\u9fff]+$"
)


def tokenize_lexical(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _LEXICAL_SEGMENT_PATTERN.finditer(text):
        segment = match.group()
        if _CJK_PATTERN.fullmatch(segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(
                    segment[index : index + 2]
                    for index in range(len(segment) - 1)
                )
            continue

        normalized = segment.casefold().strip("._-")
        if normalized:
            tokens.append(normalized)
    return tuple(tokens)


@dataclass(frozen=True, slots=True)
class Bm25Config:
    k1: float = 1.2
    b: float = 0.75
    tokenizer_version: str = "mixed-cjk-bigram-v1"

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between zero and one")
        if not self.tokenizer_version.strip():
            raise ValueError("tokenizer_version must not be blank")


@dataclass(frozen=True, slots=True)
class Bm25SearchResult:
    rank: int
    score: float
    matched_terms: tuple[str, ...]
    chunk: EvidenceChunk

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.score <= 0:
            raise ValueError("score must be positive")
        if not self.matched_terms:
            raise ValueError("matched_terms must not be empty")


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    chunk: EvidenceChunk
    term_frequencies: Counter[str]
    document_length: int


class Bm25Retriever:
    def __init__(
        self,
        chunks: list[EvidenceChunk],
        config: Bm25Config = Bm25Config(),
    ) -> None:
        evidence_ids = [chunk.evidence_id for chunk in chunks]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("corpus contains duplicate evidence IDs")

        self._config = config
        self._documents = [
            self._index_chunk(chunk) for chunk in chunks
        ]
        self._average_document_length = (
            sum(
                document.document_length
                for document in self._documents
            )
            / len(self._documents)
            if self._documents
            else 0.0
        )
        self._document_frequencies = self._build_frequencies(
            self._documents
        )

    @property
    def corpus_size(self) -> int:
        return len(self._documents)

    @property
    def tokenizer_version(self) -> str:
        return self._config.tokenizer_version

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Bm25SearchResult]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self._documents:
            return []

        query_terms = frozenset(tokenize_lexical(query))
        scored: list[
            tuple[float, tuple[str, ...], EvidenceChunk]
        ] = []
        for document in self._documents:
            matched_terms = tuple(
                sorted(
                    query_terms
                    & document.term_frequencies.keys()
                )
            )
            if not matched_terms:
                continue

            score = sum(
                self._term_score(document, term)
                for term in matched_terms
            )
            if score > 0:
                scored.append(
                    (score, matched_terms, document.chunk)
                )

        scored.sort(
            key=lambda item: (-item[0], item[2].evidence_id)
        )
        return [
            Bm25SearchResult(
                rank=rank,
                score=score,
                matched_terms=matched_terms,
                chunk=chunk,
            )
            for rank, (score, matched_terms, chunk) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]

    def _term_score(
        self,
        document: _IndexedChunk,
        term: str,
    ) -> float:
        term_frequency = document.term_frequencies[term]
        document_frequency = self._document_frequencies[term]
        corpus_size = len(self._documents)
        inverse_document_frequency = math.log(
            1
            + (
                corpus_size
                -  document_frequency
                + 0.5
            )
            / (document_frequency + 0.5)
        )
        length_ratio = (
            document.document_length
            / self._average_document_length
        )
        denominator = term_frequency + self._config.k1 * (
            1 - self._config.b
            + self._config.b * length_ratio
        )
        term_frequency_weight = (
            term_frequency * (self._config.k1 + 1)
            / denominator
        )
        return (
            inverse_document_frequency
            * term_frequency_weight
        )

    @staticmethod
    def _index_chunk(chunk: EvidenceChunk) -> _IndexedChunk:
        tokens = tokenize_lexical(chunk.text)
        return _IndexedChunk(
            chunk=chunk,
            term_frequencies=Counter(tokens),
            document_length=len(tokens),
        )

    @staticmethod
    def _build_frequencies(
        documents: list[_IndexedChunk],
    ) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for document in documents:
            frequencies.update(document.term_frequencies.keys())
        return frequencies
