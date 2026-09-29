"""Add explicit activation gate for structured job records.

Revision ID: 20260819_07
Revises: 20260819_06
Create Date: 2026-08-19
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260819_07"
down_revision: str | None = "20260819_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "structured_job_records",
        sa.Column(
            "is_current",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "structured_job_records",
        sa.Column("activated_at", sa.DateTime(timezone=True)),
    )
    op.add_column(
        "structured_job_records",
        sa.Column("activation_reference", sa.String(500)),
    )
    op.create_check_constraint(
        "ck_structured_job_records_current_activation",
        "structured_job_records",
        "NOT is_current OR "
        "(activated_at IS NOT NULL "
        "AND activation_reference IS NOT NULL)",
    )
    op.create_index(
        "uq_structured_job_records_one_current_per_snapshot",
        "structured_job_records",
        ["source_snapshot_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_structured_job_records_one_current_per_snapshot",
        table_name="structured_job_records",
    )
    op.drop_constraint(
        "ck_structured_job_records_current_activation",
        "structured_job_records",
        type_="check",
    )
    op.drop_column("structured_job_records", "activation_reference")
    op.drop_column("structured_job_records", "activated_at")
    op.drop_column("structured_job_records", "is_current")
