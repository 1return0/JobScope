from __future__ import annotations

from dataclasses import dataclass

from app.application.retrieval.current_corpus_search import CurrentCorpusReader
from app.application.retrieval.dense_retrieval import (
    DenseIndexBuilder,
    DenseSearchResult,
    InMemoryDenseRetriever,
    TextEmbedder,
)


@dataclass(frozen=True, slots=True)
class CurrentCorpusDenseSearchReport:
    query: str
    top_k: int
    corpus_size: int
    chunker_version: str
    embedding_identity: str
    corpus_fingerprint: str
    results: tuple[DenseSearchResult, ...]


class CurrentCorpusDenseSearchService:
    def __init__(
        self,
        corpus_reader: CurrentCorpusReader,
        embedder: TextEmbedder,
        *,
        chunker_version: str,
    ) -> None:
        normalized_version = chunker_version.strip()
        if not normalized_version:
            raise ValueError("chunker_version must not be blank")
        self._corpus_reader = corpus_reader
        self._embedder = embedder
        self._chunker_version = normalized_version

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_reference: str | None = None,
    ) -> CurrentCorpusDenseSearchReport:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        chunks = self._corpus_reader.list_current_chunks(
            chunker_version=self._chunker_version,
            source_reference=source_reference,
        )
        if not chunks:
            raise ValueError("current corpus must not be empty")

        index = DenseIndexBuilder(self._embedder).build(chunks)
        retriever = InMemoryDenseRetriever(index, self._embedder)
        results = tuple(
            retriever.search(normalized_query, top_k=top_k)
        )
        return CurrentCorpusDenseSearchReport(
            query=normalized_query,
            top_k=top_k,
            corpus_size=retriever.corpus_size,
            chunker_version=index.chunker_version,
            embedding_identity=index.embedding_identity,
            corpus_fingerprint=index.corpus_fingerprint,
            results=results,
        )
