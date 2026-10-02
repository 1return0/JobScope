from __future__ import annotations

import os
import unittest
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.config import load_settings
from app.infrastructure.persistence.database import build_postgresql_dsn
from app.infrastructure.postgres_checkpointer import open_postgres_checkpointer


class _CheckpointSmokeState(TypedDict, total=False):
    counter: int


def _increment(state: _CheckpointSmokeState) -> _CheckpointSmokeState:
    return {"counter": state.get("counter", 0) + 1}


@unittest.skipUnless(
    os.getenv("JOBSCOPE_RUN_POSTGRES_CHECKPOINT_SMOKE") == "true",
    "set JOBSCOPE_RUN_POSTGRES_CHECKPOINT_SMOKE=true to run PostgreSQL smoke",
)
class PostgresCheckpointerSmokeTest(unittest.TestCase):
    def test_same_thread_restores_previous_state(self) -> None:
        builder = StateGraph(_CheckpointSmokeState)
        builder.add_node("increment", _increment)
        builder.add_edge(START, "increment")
        builder.add_edge("increment", END)
        thread_id = f"jobscope-checkpoint-smoke-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        with open_postgres_checkpointer(
            build_postgresql_dsn(load_settings())
        ) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            try:
                first_state = graph.invoke({"counter": 0}, config=config)
                second_state = graph.invoke({}, config=config)

                self.assertEqual(1, first_state["counter"])
                self.assertEqual(2, second_state["counter"])
                self.assertIsNotNone(checkpointer.get(config))
            finally:
                checkpointer.delete_thread(thread_id)


if __name__ == "__main__":
    unittest.main()
