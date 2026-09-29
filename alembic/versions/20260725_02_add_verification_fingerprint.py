"""Add deterministic verification fingerprint.

Revision ID: 20260725_02
Revises: 20260725_01
Create Date: 2026-07-25
"""

import hashlib
from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260725_02"
down_revision: str | None = "20260725_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _fingerprint(row: sa.Row) -> str:
    source = "\x1f".join(
        [
            row.company.strip().casefold(),
            row.domain.strip().lower().rstrip("."),
            row.evidence_reference.strip(),
            row.verified_at.isoformat(),
            row.verified_by.strip().casefold(),
            row.status,
        ]
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def upgrade() -> None:
    op.add_column(
        "verified_company_domains",
        sa.Column(
            "record_fingerprint",
            sa.String(64),
            nullable=True,
        ),
    )

    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT id, company, domain, evidence_reference, "
            "verified_at, verified_by, status "
            "FROM verified_company_domains"
        )
    ).all()
    for row in rows:
        connection.execute(
            sa.text(
                "UPDATE verified_company_domains "
                "SET record_fingerprint = :fingerprint "
                "WHERE id = :row_id"
            ),
            {
                "fingerprint": _fingerprint(row),
                "row_id": row.id,
            },
        )

    op.alter_column(
        "verified_company_domains",
        "record_fingerprint",
        existing_type=sa.String(64),
        nullable=False,
    )
    op.create_unique_constraint(
        "uq_verified_company_domains_fingerprint",
        "verified_company_domains",
        ["record_fingerprint"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_verified_company_domains_fingerprint",
        "verified_company_domains",
        type_="unique",
    )
    op.drop_column(
        "verified_company_domains",
        "record_fingerprint",
    )
