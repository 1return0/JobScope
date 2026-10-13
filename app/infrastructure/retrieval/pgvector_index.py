"""Small exact-search pgvector backend; no ANN quality claims or silent fallback.

Provision the vector extension separately. Snapshot replacement is one transaction.
Each application/index name has a locked pointer; readers reject stale snapshots.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from contextlib import contextmanager

import psycopg
from psycopg.types.json import Jsonb


class PgvectorIndexUnavailableError(RuntimeError):
    pass


def vector_literal(values, dimension: int) -> str:
    try:
        vector = tuple(struct.unpack("f", struct.pack("f", float(value)))[0] for value in values)
    except (OverflowError, TypeError) as error:
        raise ValueError("vector is outside float32 range") from error
    if dimension <= 0 or len(vector) != dimension:
        raise ValueError("vector dimension mismatch")
    if not all(math.isfinite(value) for value in vector) or not any(vector):
        raise ValueError("vector must be finite and non-zero")
    return "[" + ",".join(str(value) for value in vector) + "]"


class PgvectorIndex:
    def __init__(self, dsn: str, *, index_name: str) -> None:
        if not dsn.strip() or not index_name.strip():
            raise ValueError("pgvector requires a DSN and index name")
        self._dsn, self._name = dsn, index_name

    @contextmanager
    def _connection(self):
        try:
            with psycopg.connect(self._dsn, connect_timeout=5) as connection:
                connection.execute("SET LOCAL statement_timeout = '10s'")
                yield connection
        except psycopg.Error as error:
            # Do not disclose connection strings, database queries or payloads.
            raise PgvectorIndexUnavailableError("pgvector index is unavailable") from error

    def setup(self) -> None:
        """Creates only application-owned tables, not extensions or other databases."""
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS agent_dense_index (
                index_name text PRIMARY KEY, snapshot_id text NOT NULL,
                metadata jsonb NOT NULL, dimension integer NOT NULL CHECK (dimension > 0)
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS agent_dense_vector (
                index_name text NOT NULL REFERENCES agent_dense_index(index_name),
                chunk_id text NOT NULL, embedding vector NOT NULL,
                PRIMARY KEY (index_name, chunk_id)
            )""")

    def load(self):
        with self._connection() as connection:
            # SHARE prevents concurrent replacement between metadata and vectors.
            row = connection.execute("""SELECT snapshot_id, metadata, dimension
                FROM agent_dense_index WHERE index_name = %s FOR SHARE""", (self._name,)).fetchone()
            if row is None:
                return None
            vectors = connection.execute("""SELECT chunk_id, embedding::text
                FROM agent_dense_vector WHERE index_name = %s ORDER BY chunk_id""", (self._name,)).fetchall()
            decoded = tuple((key, tuple(json.loads(value))) for key, value in vectors)
            if self.snapshot_id(row[1], decoded, row[2]) != row[0]:
                raise PgvectorIndexUnavailableError("pgvector snapshot integrity mismatch")
            return row[1], decoded, row[2], row[0]

    @staticmethod
    def snapshot_id(metadata, vectors, dimension) -> str:
        # pgvector stores float32, so canonicalize before hashing and saving.
        canonical = tuple(sorted((key, tuple(struct.unpack("f", struct.pack("f", float(v)))[0]
            for v in values)) for key, values in vectors))
        payload = json.dumps([metadata, canonical, dimension], sort_keys=True,
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, metadata, vectors, dimension: int) -> str:
        vectors = tuple(vectors)
        if len({key for key, _ in vectors}) != len(vectors):
            raise ValueError("duplicate vector chunk IDs")
        literals = tuple((key, vector_literal(values, dimension)) for key, values in vectors)
        snapshot_id = self.snapshot_id(metadata, vectors, dimension)
        with self._connection() as connection:
            connection.execute("""INSERT INTO agent_dense_index VALUES (%s,%s,%s,%s)
                ON CONFLICT (index_name) DO UPDATE SET snapshot_id=EXCLUDED.snapshot_id,
                metadata=EXCLUDED.metadata, dimension=EXCLUDED.dimension""",
                (self._name, snapshot_id, Jsonb(metadata), dimension))
            connection.execute("DELETE FROM agent_dense_vector WHERE index_name=%s", (self._name,))
            with connection.cursor() as cursor:
                cursor.executemany("""INSERT INTO agent_dense_vector
                    (index_name,chunk_id,embedding) VALUES (%s,%s,%s::vector)""",
                    ((self._name, key, literal) for key, literal in literals))
        return snapshot_id

    def search(self, query_vector, *, dimension: int, snapshot_id: str, top_k: int):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        literal = vector_literal(query_vector, dimension)
        with self._connection() as connection:
            row = connection.execute("""SELECT snapshot_id, dimension FROM agent_dense_index
                WHERE index_name=%s FOR SHARE""", (self._name,)).fetchone()
            if row != (snapshot_id, dimension):
                raise PgvectorIndexUnavailableError("pgvector snapshot is stale; prepare again")
            return tuple(connection.execute("""SELECT chunk_id,
                1 - (embedding <=> %s::vector) AS score
                FROM agent_dense_vector WHERE index_name=%s
                ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s""",
                (literal, self._name, literal, top_k)).fetchall())
