from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.application.retrieval.lexical_retrieval import (
    Bm25Config,
    Bm25Retriever,
    Bm25SearchResult,
)
from app.domain.documents.document_chunking import EvidenceChunk


class CurrentCorpusReader(Protocol):
    def list_current_chunks(
        self,
        *,
        chunker_version: str,
        source_reference: str | None = None,
    ) -> list[EvidenceChunk]: ...


@dataclass(frozen=True, slots=True)
class CurrentCorpusSearchReport:
    query: str
    top_k: int
    corpus_size: int
    chunker_version: str
    tokenizer_version: str
    results: tuple[Bm25SearchResult, ...]


class CurrentCorpusSearchService:
    def __init__(
        self,
        corpus_reader: CurrentCorpusReader,
        *,
        chunker_version: str,
        bm25_config: Bm25Config = Bm25Config(),
    ) -> None:
        normalized_version = chunker_version.strip()
        if not normalized_version:
            raise ValueError("chunker_version must not be blank")
        self._corpus_reader = corpus_reader
        self._chunker_version = normalized_version
        self._bm25_config = bm25_config

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_reference: str | None = None,
    ) -> CurrentCorpusSearchReport:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        chunks = self._corpus_reader.list_current_chunks(
            chunker_version=self._chunker_version,
            source_reference=source_reference,
        )
        retriever = Bm25Retriever(chunks, self._bm25_config)
        results = tuple(
            retriever.search(normalized_query, top_k=top_k)
        )
        return CurrentCorpusSearchReport(
            query=normalized_query,
            top_k=top_k,
            corpus_size=retriever.corpus_size,
            chunker_version=self._chunker_version,
            tokenizer_version=retriever.tokenizer_version,
            results=results,
        )
