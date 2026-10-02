"""Create opaque server-side Job Agent sessions.

Revision ID: 20260930_08
Revises: 20260819_07
Create Date: 2026-09-30
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260930_08"
down_revision: str | None = "20260819_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "job_agent_sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("owner_id", sa.String(36), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("owner_id", name="uq_job_agent_sessions_owner"),
    )
    op.create_index(
        "ix_job_agent_sessions_expires_at",
        "job_agent_sessions",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_job_agent_sessions_expires_at",
        table_name="job_agent_sessions",
    )
    op.drop_table("job_agent_sessions")
