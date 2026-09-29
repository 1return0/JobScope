"""Link document snapshots and evidence chunks.

Revision ID: 20260728_05
Revises: 20260728_04
Create Date: 2026-07-28
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260728_05"
down_revision: str | None = "20260728_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_snapshot_chunks",
        sa.Column(
            "snapshot_id",
            sa.String(69),
            sa.ForeignKey(
                "document_snapshots.snapshot_id",
                ondelete="CASCADE",
            ),
            primary_key=True,
        ),
        sa.Column(
            "evidence_id",
            sa.String(67),
            sa.ForeignKey(
                "evidence_chunks.evidence_id",
                ondelete="CASCADE",
            ),
            primary_key=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "evidence_id",
            name="uq_document_snapshot_chunks_evidence",
        ),
    )
    op.create_index(
        "ix_document_snapshot_chunks_snapshot",
        "document_snapshot_chunks",
        ["snapshot_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_snapshot_chunks_snapshot",
        table_name="document_snapshot_chunks",
    )
    op.drop_table("document_snapshot_chunks")
