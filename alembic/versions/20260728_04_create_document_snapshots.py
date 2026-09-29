"""Create versioned document snapshots.

Revision ID: 20260728_04
Revises: 20260727_03
Create Date: 2026-07-28
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260728_04"
down_revision: str | None = "20260727_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_snapshots",
        sa.Column(
            "snapshot_id",
            sa.String(69),
            primary_key=True,
        ),
        sa.Column(
            "source_reference",
            sa.String(2048),
            nullable=False,
        ),
        sa.Column(
            "document_format",
            sa.String(16),
            nullable=False,
        ),
        sa.Column(
            "content_sha256",
            sa.String(64),
            nullable=False,
        ),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column(
            "version_number",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "first_observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "last_observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "document_format IN ('html', 'pdf', 'docx', 'xlsx', 'pptx')",
            name="ck_document_snapshots_format",
        ),
        sa.CheckConstraint(
            "byte_size >= 0",
            name="ck_document_snapshots_byte_size",
        ),
        sa.CheckConstraint(
            "version_number >= 1",
            name="ck_document_snapshots_version_number",
        ),
        sa.CheckConstraint(
            "last_observed_at >= first_observed_at",
            name="ck_document_snapshots_observation_order",
        ),
        sa.UniqueConstraint(
            "source_reference",
            "content_sha256",
            name="uq_document_snapshots_source_content",
        ),
        sa.UniqueConstraint(
            "source_reference",
            "version_number",
            name="uq_document_snapshots_source_version",
        ),
    )
    op.create_index(
        "ix_document_snapshots_current",
        "document_snapshots",
        ["source_reference", "is_current"],
    )
    op.create_index(
        "uq_document_snapshots_one_current",
        "document_snapshots",
        ["source_reference"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_document_snapshots_one_current",
        table_name="document_snapshots",
    )
    op.drop_index(
        "ix_document_snapshots_current",
        table_name="document_snapshots",
    )
    op.drop_table("document_snapshots")
