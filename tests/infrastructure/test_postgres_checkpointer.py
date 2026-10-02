from __future__ import annotations

import unittest
from unittest.mock import ANY, MagicMock, patch

from app.infrastructure.postgres_checkpointer import open_postgres_checkpointer


class PostgresCheckpointerTest(unittest.TestCase):
    def test_blank_dsn_is_rejected_before_opening_a_connection(self) -> None:
        with self.assertRaisesRegex(ValueError, "DSN"):
            with open_postgres_checkpointer("   "):
                pass

    @patch("app.infrastructure.postgres_checkpointer.PostgresSaver")
    @patch("app.infrastructure.postgres_checkpointer.psycopg.connect")
    def test_prepares_saver_with_safe_connection_options(
        self,
        connect: MagicMock,
        saver_type: MagicMock,
    ) -> None:
        saver = saver_type.return_value

        with open_postgres_checkpointer("postgresql://example") as actual:
            self.assertIs(saver, actual)

        connect.assert_called_once_with(
            "postgresql://example",
            autocommit=True,
            row_factory=ANY,
            prepare_threshold=0,
        )
        saver.setup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
