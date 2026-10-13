from __future__ import annotations

from app.application.retrieval.dense_retrieval import (
    DenseIndexBuilder, DenseSearchResult, _build_corpus_fingerprint, _validate_vector,
)
from app.infrastructure.retrieval.pgvector_index import PgvectorIndex, PgvectorIndexUnavailableError


class PgvectorDenseRetriever:
    """Exact vector distance in PostgreSQL; chunks/citations stay in reviewed corpus."""
    def __init__(self, chunks, embedder, *, dsn: str, index_name="jobscope-current-corpus"):
        if not chunks:
            raise ValueError("pgvector requires chunks")
        self._chunks = {chunk.evidence_id: chunk for chunk in chunks}
        if len(self._chunks) != len(chunks):
            raise ValueError("duplicate evidence IDs")
        if len({chunk.chunker_version for chunk in chunks}) != 1:
            raise ValueError("mixed chunker versions")
        self._embedder = embedder
        self._fingerprint = _build_corpus_fingerprint(sorted(chunks, key=lambda c: c.evidence_id))
        metadata = {"corpus_fingerprint": self._fingerprint,
                    "embedding_identity": embedder.spec.identity,
                    "chunker_version": chunks[0].chunker_version}
        self._index = PgvectorIndex(dsn, index_name=index_name)
        self._index.setup()
        existing = self._index.load()
        if (existing is not None and existing[0] == metadata
                and existing[2] == embedder.spec.dimension
                and {key for key, _ in existing[1]} == set(self._chunks)):
            self._snapshot_id = existing[3]
        else:
            artifact = DenseIndexBuilder(embedder).build(chunks)
            self._snapshot_id = self._index.save(metadata,
                ((row.chunk.evidence_id, row.vector) for row in artifact.records),
                artifact.dimension)

    @property
    def corpus_size(self):
        return len(self._chunks)

    @property
    def corpus_fingerprint(self):
        return self._fingerprint

    def search(self, query, *, top_k=5):
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = _validate_vector(self._embedder.embed_query(query.strip()),
            dimension=self._embedder.spec.dimension, normalization=self._embedder.spec.normalization)
        rows = self._index.search(vector, dimension=self._embedder.spec.dimension,
                                  snapshot_id=self._snapshot_id, top_k=top_k)
        if any(key not in self._chunks for key, _ in rows):
            raise PgvectorIndexUnavailableError("pgvector returned unknown evidence")
        return [DenseSearchResult(rank, score, self._chunks[key])
                for rank, (key, score) in enumerate(rows, start=1)]
