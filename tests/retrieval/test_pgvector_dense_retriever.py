import unittest
from unittest.mock import patch

from app.infrastructure.retrieval.pgvector_index import PgvectorIndex, vector_literal
from app.infrastructure.retrieval.pgvector_dense_retriever import PgvectorDenseRetriever
from app.application.retrieval.dense_retrieval import _build_corpus_fingerprint
from tests.retrieval.test_current_corpus_hybrid_search import _Embedder
from tests.retrieval import test_current_corpus_hybrid_search as fixtures


class PgvectorDenseTest(unittest.TestCase):
    def test_reuses_existing_vectors_without_document_embedding(self):
        chunk = fixtures.CurrentCorpusHybridSearchServiceTest._chunk(1, "school selection")
        embedder = _Embedder()
        metadata = {"corpus_fingerprint": _build_corpus_fingerprint([chunk]),
            "embedding_identity": embedder.spec.identity, "chunker_version": chunk.chunker_version}
        with patch("app.infrastructure.retrieval.pgvector_dense_retriever.PgvectorIndex") as factory:
            backend = factory.return_value
            backend.load.return_value = (metadata, ((chunk.evidence_id, (1, 0)),), 2, "snapshot-key")
            backend.search.return_value = ((chunk.evidence_id, 1.0),)
            retriever = PgvectorDenseRetriever([chunk], embedder, dsn="postgresql://test.invalid/db")
            result = retriever.search("school", top_k=1)
            self.assertEqual(0, embedder.document_calls)
            self.assertEqual(chunk, result[0].chunk)
            backend.search.assert_called_once_with((1.0, 0.0), dimension=2,
                snapshot_id="snapshot-key", top_k=1)

    def test_validation_and_snapshot_identity(self):
        with self.assertRaises(ValueError):
            vector_literal((float("nan"), 1), 2)
        self.assertNotEqual(PgvectorIndex.snapshot_id({"model": "a"}, (("id", (1, 0)),), 2),
                            PgvectorIndex.snapshot_id({"model": "b"}, (("id", (1, 0)),), 2))
