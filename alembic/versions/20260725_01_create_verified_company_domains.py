"""Create verified company domain history.

Revision ID: 20260725_01
Revises:
Create Date: 2026-07-25
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260725_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "verified_company_domains",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("company", sa.String(200), nullable=False),
        sa.Column(
            "normalized_company",
            sa.String(200),
            nullable=False,
        ),
        sa.Column("domain", sa.String(253), nullable=False),
        sa.Column(
            "normalized_domain",
            sa.String(253),
            nullable=False,
        ),
        sa.Column(
            "evidence_reference",
            sa.String(500),
            nullable=False,
        ),
        sa.Column("verified_at", sa.Date(), nullable=False),
        sa.Column("verified_by", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'revoked')",
            name="ck_verified_company_domains_status",
        ),
    )
    op.create_index(
        "ix_verified_company_domains_lookup",
        "verified_company_domains",
        [
            "normalized_company",
            "normalized_domain",
            "verified_at",
        ],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_verified_company_domains_lookup",
        table_name="verified_company_domains",
    )
    op.drop_table("verified_company_domains")
