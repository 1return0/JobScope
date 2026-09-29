"""Create evidence chunk storage.

Revision ID: 20260727_03
Revises: 20260725_02
Create Date: 2026-07-27
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260727_03"
down_revision: str | None = "20260725_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evidence_chunks",
        sa.Column(
            "evidence_id",
            sa.String(67),
            primary_key=True,
        ),
        sa.Column(
            "document_sha256",
            sa.String(64),
            nullable=False,
        ),
        sa.Column(
            "source_reference",
            sa.String(2048),
            nullable=False,
        ),
        sa.Column(
            "source_fragment_ordinal",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "chunk_ordinal",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "chunker_version",
            sa.String(200),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("heading_path", sa.JSON(), nullable=False),
        sa.Column("sheet_name", sa.String(255), nullable=True),
        sa.Column("cell_reference", sa.String(100), nullable=True),
        sa.Column("slide_number", sa.Integer(), nullable=True),
        sa.Column("table_number", sa.Integer(), nullable=True),
        sa.Column("table_row_number", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_fragment_ordinal >= 0",
            name="ck_evidence_chunks_source_fragment_ordinal",
        ),
        sa.CheckConstraint(
            "chunk_ordinal >= 0",
            name="ck_evidence_chunks_chunk_ordinal",
        ),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="ck_evidence_chunks_page_number",
        ),
        sa.CheckConstraint(
            "slide_number IS NULL OR slide_number >= 1",
            name="ck_evidence_chunks_slide_number",
        ),
        sa.CheckConstraint(
            "table_number IS NULL OR table_number >= 1",
            name="ck_evidence_chunks_table_number",
        ),
        sa.CheckConstraint(
            (
                "table_row_number IS NULL "
                "OR table_row_number >= 1"
            ),
            name="ck_evidence_chunks_table_row_number",
        ),
        sa.CheckConstraint(
            (
                "table_row_number IS NULL "
                "OR table_number IS NOT NULL"
            ),
            name="ck_evidence_chunks_table_row_requires_table",
        ),
    )
    op.create_index(
        "ix_evidence_chunks_document",
        "evidence_chunks",
        ["document_sha256", "chunker_version"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_evidence_chunks_document",
        table_name="evidence_chunks",
    )
    op.drop_table("evidence_chunks")
