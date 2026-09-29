"""Create structured job records and citation lineage.

Revision ID: 20260819_06
Revises: 20260728_05
Create Date: 2026-08-19
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260819_06"
down_revision: str | None = "20260728_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "structured_job_records",
        sa.Column("record_id", sa.String(68), primary_key=True),
        sa.Column("record_fingerprint", sa.String(64), nullable=False),
        sa.Column(
            "source_snapshot_id",
            sa.String(69),
            sa.ForeignKey(
                "document_snapshots.snapshot_id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "extraction_contract_version",
            sa.String(200),
            nullable=False,
        ),
        sa.Column("prompt_version", sa.String(200), nullable=False),
        sa.Column("model_identity", sa.Text(), nullable=False),
        sa.Column("judge_identity", sa.Text(), nullable=False),
        sa.Column("record_payload", sa.JSON(), nullable=False),
        sa.Column("company", sa.String(500)),
        sa.Column("job_title", sa.String(500)),
        sa.Column("locations", sa.JSON(), nullable=False),
        sa.Column("education_requirement", sa.Text()),
        sa.Column("major_requirement", sa.Text()),
        sa.Column("recruitment_type", sa.String(200)),
        sa.Column("application_deadline_raw", sa.Text()),
        sa.Column("application_deadline_normalized", sa.Date()),
        sa.Column(
            "deadline_normalization_status",
            sa.String(32),
            nullable=False,
        ),
        sa.Column("deadline_reason_code", sa.String(100)),
        sa.Column(
            "date_normalizer_version",
            sa.String(100),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "deadline_normalization_status IN "
            "('normalized', 'not_stated', 'manual_review_required')",
            name="ck_structured_job_records_deadline_status",
        ),
        sa.CheckConstraint(
            "(deadline_normalization_status = 'normalized' "
            "AND application_deadline_normalized IS NOT NULL "
            "AND deadline_reason_code IS NULL) OR "
            "(deadline_normalization_status = 'not_stated' "
            "AND application_deadline_raw IS NULL "
            "AND application_deadline_normalized IS NULL "
            "AND deadline_reason_code IS NULL) OR "
            "(deadline_normalization_status = 'manual_review_required' "
            "AND application_deadline_raw IS NOT NULL "
            "AND application_deadline_normalized IS NULL "
            "AND deadline_reason_code IS NOT NULL)",
            name="ck_structured_job_records_deadline_consistency",
        ),
        sa.UniqueConstraint(
            "record_fingerprint",
            name="uq_structured_job_records_fingerprint",
        ),
    )
    op.create_index(
        "ix_structured_job_records_snapshot",
        "structured_job_records",
        ["source_snapshot_id"],
    )
    op.create_index(
        "ix_structured_job_records_deadline",
        "structured_job_records",
        ["application_deadline_normalized"],
    )

    op.create_table(
        "structured_job_record_citations",
        sa.Column(
            "record_id",
            sa.String(68),
            sa.ForeignKey(
                "structured_job_records.record_id",
                ondelete="CASCADE",
            ),
            primary_key=True,
        ),
        sa.Column("field_path", sa.String(200), primary_key=True),
        sa.Column("citation_ordinal", sa.Integer(), primary_key=True),
        sa.Column(
            "evidence_id",
            sa.String(67),
            sa.ForeignKey(
                "evidence_chunks.evidence_id",
                ondelete="RESTRICT",
            ),
            nullable=False,
        ),
        sa.Column("quoted_text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "citation_ordinal >= 1",
            name="ck_structured_job_record_citations_ordinal",
        ),
    )
    op.create_index(
        "ix_structured_job_record_citations_evidence",
        "structured_job_record_citations",
        ["evidence_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_structured_job_record_citations_evidence",
        table_name="structured_job_record_citations",
    )
    op.drop_table("structured_job_record_citations")
    op.drop_index(
        "ix_structured_job_records_deadline",
        table_name="structured_job_records",
    )
    op.drop_index(
        "ix_structured_job_records_snapshot",
        table_name="structured_job_records",
    )
    op.drop_table("structured_job_records")
