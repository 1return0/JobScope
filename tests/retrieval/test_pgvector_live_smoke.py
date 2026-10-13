"""Opt-in PostgreSQL/vector smoke. Only a fresh private schema is mutated."""
import os
import re
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app.infrastructure.retrieval.pgvector_index import PgvectorIndex, PgvectorIndexUnavailableError


class SecretDsn(str):
    def __repr__(self):
        return "<private pgvector smoke DSN>"


@pytest.fixture
def vector_dsn():
    dsn = os.getenv("AGENT_PGVECTOR_SMOKE_DSN")
    if not dsn:
        pytest.skip("explicit AGENT_PGVECTOR_SMOKE_DSN required")
    schema = "agent_vector_smoke_" + uuid.uuid4().hex
    assert re.fullmatch(r"agent_vector_smoke_[0-9a-f]{32}", schema)
    with psycopg.connect(dsn, autocommit=True) as admin:
        if admin.execute("SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone() is None:
            pytest.fail("pgvector extension not provisioned; this is not a passing smoke")
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield SecretDsn(make_conninfo(dsn, options=f"-c search_path={schema},public"))
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            # Exact, validated namespace created by this fixture, never public.
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_real_database_cosine_restart_and_index_isolation(vector_dsn):
    index = PgvectorIndex(vector_dsn, index_name="test-one")
    index.setup()
    key = index.save({"model": "fake-v1"}, (("a", (1, 0)), ("b", (0, 1))), 2)
    restarted = PgvectorIndex(vector_dsn, index_name="test-one")
    assert restarted.load()[3] == key
    rows = restarted.search((1, 0), dimension=2, snapshot_id=key, top_k=2)
    assert [row[0] for row in rows] == ["a", "b"]
    assert rows[0][1] == pytest.approx(1)
    assert rows[1][1] == pytest.approx(0)
    other = PgvectorIndex(vector_dsn, index_name="test-two")
    other_key = other.save({"model": "fake-v1"}, (("foreign", (1, 0)),), 2)
    assert other.search((1, 0), dimension=2, snapshot_id=other_key, top_k=5)[0][0] == "foreign"
    assert restarted.search((1, 0), dimension=2, snapshot_id=key, top_k=1)[0][0] == "a"


def test_failed_snapshot_transaction_keeps_old_vectors(vector_dsn):
    index = PgvectorIndex(vector_dsn, index_name="test-rollback")
    index.setup()
    key = index.save({"model": "old"}, (("good", (1, 0)),), 2)
    with psycopg.connect(vector_dsn) as connection:
        connection.execute("ALTER TABLE agent_dense_vector ADD CONSTRAINT test_failure CHECK(chunk_id <> 'broken')")
    with pytest.raises(PgvectorIndexUnavailableError):
        index.save({"model": "new"}, (("broken", (0, 1)),), 2)
    assert index.load()[3] == key
    assert index.search((1, 0), dimension=2, snapshot_id=key, top_k=1)[0][0] == "good"


def test_old_reader_rejects_replaced_snapshot(vector_dsn):
    index = PgvectorIndex(vector_dsn, index_name="test-stale")
    index.setup()
    old_key = index.save({"model": "old"}, (("a", (1, 0)),), 2)
    index.save({"model": "new"}, (("a", (0, 1)),), 2)
    with pytest.raises(PgvectorIndexUnavailableError, match="stale"):
        index.search((1, 0), dimension=2, snapshot_id=old_key, top_k=1)


def test_real_hybrid_adapter_reuses_vectors_and_preserves_source_scope(vector_dsn):
    from app.application.retrieval.current_corpus_hybrid_search import CurrentCorpusHybridSearchService
    from app.infrastructure.retrieval.pgvector_dense_retriever import PgvectorDenseRetriever
    from tests.retrieval import test_current_corpus_hybrid_search as fixtures

    chunks = [fixtures.CurrentCorpusHybridSearchServiceTest._chunk(1, "院校信息", "source-a"),
              fixtures.CurrentCorpusHybridSearchServiceTest._chunk(2, "library", "source-b")]
    embedder = fixtures._Embedder()
    factory = lambda rows, model: PgvectorDenseRetriever(rows, model,
        dsn=vector_dsn, index_name="jobscope-adapter-smoke")
    def service():
        return CurrentCorpusHybridSearchService(fixtures._Reader(chunks), embedder,
            chunker_version="structure-v1", candidate_k=2, rank_constant=60,
            dense_factory=factory)
    first = service()
    first.prepare()
    assert first.search("school", top_k=1).results[0].chunk.evidence_id == chunks[0].evidence_id
    restarted = service()
    restarted.prepare()
    scoped = restarted.search("school", top_k=1, source_references=("source-b",))
    assert scoped.corpus_size == 1
    assert scoped.results[0].chunk.source_reference == "source-b"
    assert embedder.document_calls == 1
