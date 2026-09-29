from __future__ import annotations

from dataclasses import dataclass
from threading import BoundedSemaphore, Lock

from app.application.retrieval.current_corpus_search import CurrentCorpusReader
from app.application.retrieval.dense_retrieval import (
    DenseIndexBuilder,
    InMemoryDenseRetriever,
    TextEmbedder,
)
from app.application.retrieval.lexical_retrieval import Bm25Config, Bm25Retriever
from app.application.retrieval.reciprocal_rank_fusion import (
    FusedEvidenceRank,
    reciprocal_rank_fusion,
)
from app.domain.documents.document_chunking import EvidenceChunk


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    rank: int
    fusion_score: float
    ranks_by_retriever: tuple[tuple[str, int], ...]
    chunk: EvidenceChunk


class HybridSearchCapacityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CurrentCorpusHybridSearchReport:
    query: str
    top_k: int
    candidate_k: int
    rank_constant: int
    corpus_size: int
    chunker_version: str
    tokenizer_version: str
    embedding_identity: str
    corpus_fingerprint: str
    results: tuple[HybridSearchResult, ...]


@dataclass(frozen=True, slots=True)
class _PreparedHybridState:
    bm25: Bm25Retriever
    dense: InMemoryDenseRetriever
    chunks_by_id: dict[str, EvidenceChunk]


class CurrentCorpusHybridSearchService:
    def __init__(
        self,
        corpus_reader: CurrentCorpusReader,
        embedder: TextEmbedder,
        *,
        chunker_version: str,
        candidate_k: int,
        rank_constant: int,
        max_concurrent_queries: int = 1,
        bm25_config: Bm25Config = Bm25Config(),
    ) -> None:
        normalized_version = chunker_version.strip()
        if not normalized_version:
            raise ValueError("chunker_version must not be blank")
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if rank_constant < 1:
            raise ValueError("rank_constant must be positive")
        if max_concurrent_queries < 1:
            raise ValueError("max_concurrent_queries must be positive")
        self._corpus_reader = corpus_reader
        self._embedder = embedder
        self._chunker_version = normalized_version
        self._candidate_k = candidate_k
        self._rank_constant = rank_constant
        self._bm25_config = bm25_config
        self._query_slots = BoundedSemaphore(max_concurrent_queries)
        self._state: _PreparedHybridState | None = None
        self._prepare_lock = Lock()

    def prepare(
        self,
        *,
        source_reference: str | None = None,
    ) -> None:
        with self._prepare_lock:
            chunks = self._corpus_reader.list_current_chunks(
                chunker_version=self._chunker_version,
                source_reference=source_reference,
            )
            if not chunks:
                raise ValueError("current corpus must not be empty")
            bm25 = Bm25Retriever(chunks, self._bm25_config)
            dense_index = DenseIndexBuilder(self._embedder).build(chunks)
            new_state = _PreparedHybridState(
                bm25=bm25,
                dense=InMemoryDenseRetriever(
                    dense_index,
                    self._embedder,
                ),
                chunks_by_id={
                    chunk.evidence_id: chunk for chunk in chunks
                },
            )
            self._state = new_state

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_references: tuple[str, ...] | None = None,
    ) -> CurrentCorpusHybridSearchReport:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if top_k > self._candidate_k:
            raise ValueError("top_k must not exceed candidate_k")
        state = self._state
        if state is None:
            raise RuntimeError("hybrid search service is not prepared")

        if not self._query_slots.acquire(blocking=False):
            raise HybridSearchCapacityError(
                "hybrid search capacity is temporarily exhausted"
            )

        try:
            return self._search_prepared(
                normalized_query,
                top_k=top_k,
                state=state,
                source_references=source_references,
            )
        finally:
            self._query_slots.release()

    def _search_prepared(
        self,
        query: str,
        *,
        top_k: int,
        state: _PreparedHybridState,
        source_references: tuple[str, ...] | None,
    ) -> CurrentCorpusHybridSearchReport:
        allowed_evidence_ids: set[str] | None = None
        search_depth = self._candidate_k
        if source_references is not None:
            normalized_sources = {
                source.strip()
                for source in source_references
                if source.strip()
            }
            allowed_evidence_ids = {
                evidence_id
                for evidence_id, chunk in state.chunks_by_id.items()
                if chunk.source_reference in normalized_sources
            }
            if not allowed_evidence_ids:
                return self._build_report(
                    query,
                    (),
                    state=state,
                    corpus_size=0,
                )
            # Rank the prepared corpus once, then retain only evidence from the
            # structured-job sources selected by the SQL tool. This reuses the
            # existing indexes and prevents unrelated documents entering the
            # mixed-tool answer.
            search_depth = len(state.chunks_by_id)
        bm25_ids = tuple(
            result.chunk.evidence_id
            for result in state.bm25.search(
                query,
                top_k=search_depth,
            )
            if allowed_evidence_ids is None
            or result.chunk.evidence_id in allowed_evidence_ids
        )
        dense_ids = tuple(
            result.chunk.evidence_id
            for result in state.dense.search(
                query,
                top_k=search_depth,
            )
            if allowed_evidence_ids is None
            or result.chunk.evidence_id in allowed_evidence_ids
        )
        bm25_ids = bm25_ids[: self._candidate_k]
        dense_ids = dense_ids[: self._candidate_k]
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_ids, "dense": dense_ids},
            top_k=top_k,
            rank_constant=self._rank_constant,
        )
        return self._build_report(
            query,
            fused,
            state=state,
            corpus_size=(
                len(allowed_evidence_ids)
                if allowed_evidence_ids is not None
                else state.dense.corpus_size
            ),
        )

    def _build_report(
        self,
        query: str,
        fused: tuple[FusedEvidenceRank, ...],
        *,
        state: _PreparedHybridState,
        corpus_size: int | None = None,
    ) -> CurrentCorpusHybridSearchReport:
        return CurrentCorpusHybridSearchReport(
            query=query,
            top_k=len(fused),
            candidate_k=self._candidate_k,
            rank_constant=self._rank_constant,
            corpus_size=(
                state.dense.corpus_size
                if corpus_size is None
                else corpus_size
            ),
            chunker_version=self._chunker_version,
            tokenizer_version=state.bm25.tokenizer_version,
            embedding_identity=self._embedder.spec.identity,
            corpus_fingerprint=state.dense.corpus_fingerprint,
            results=tuple(
                HybridSearchResult(
                    rank=item.rank,
                    fusion_score=item.fusion_score,
                    ranks_by_retriever=item.ranks_by_retriever,
                    chunk=state.chunks_by_id[item.evidence_id],
                )
                for item in fused
            ),
        )
