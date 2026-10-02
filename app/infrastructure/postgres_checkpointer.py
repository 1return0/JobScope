"""PostgreSQL-backed LangGraph checkpoints, owned by the application lifespan."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row


@contextmanager
def open_postgres_checkpointer(dsn: str) -> Iterator[PostgresSaver]:
    """Prepare a saver and close its database connection on context exit.

    Only plain, non-secret Graph state should be checkpointed. Thread ownership
    must be checked by the caller before using a thread_id with this saver.
    """
    if not dsn.strip():
        raise ValueError("checkpoint PostgreSQL DSN must not be blank")

    with psycopg.connect(
        dsn,
        autocommit=True,
        row_factory=dict_row,
        prepare_threshold=0,
    ) as connection:
        saver = PostgresSaver(
            connection,
            serde=JsonPlusSerializer(allowed_msgpack_modules=[]),
        )
        saver.setup()
        yield saver
