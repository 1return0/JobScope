from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.persistence.database import Base


class VerifiedCompanyDomainRow(Base):
    __tablename__ = "verified_company_domains"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'revoked')",
            name="ck_verified_company_domains_status",
        ),
        Index(
            "ix_verified_company_domains_lookup",
            "normalized_company",
            "normalized_domain",
            "verified_at",
        ),
        UniqueConstraint(
            "record_fingerprint",
            name="uq_verified_company_domains_fingerprint",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    record_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    company: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_company: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    domain: Mapped[str] = mapped_column(String(253), nullable=False)
    normalized_domain: Mapped[str] = mapped_column(
        String(253),
        nullable=False,
    )
    evidence_reference: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    verified_at: Mapped[date] = mapped_column(Date, nullable=False)
    verified_by: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class EvidenceChunkRow(Base):
    __tablename__ = "evidence_chunks"
    __table_args__ = (
        CheckConstraint(
            "source_fragment_ordinal >= 0",
            name="ck_evidence_chunks_source_fragment_ordinal",
        ),
        CheckConstraint(
            "chunk_ordinal >= 0",
            name="ck_evidence_chunks_chunk_ordinal",
        ),
        CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="ck_evidence_chunks_page_number",
        ),
        CheckConstraint(
            "slide_number IS NULL OR slide_number >= 1",
            name="ck_evidence_chunks_slide_number",
        ),
        CheckConstraint(
            "table_number IS NULL OR table_number >= 1",
            name="ck_evidence_chunks_table_number",
        ),
        CheckConstraint(
            (
                "table_row_number IS NULL "
                "OR table_row_number >= 1"
            ),
            name="ck_evidence_chunks_table_row_number",
        ),
        CheckConstraint(
            (
                "table_row_number IS NULL "
                "OR table_number IS NOT NULL"
            ),
            name="ck_evidence_chunks_table_row_requires_table",
        ),
        Index(
            "ix_evidence_chunks_document",
            "document_sha256",
            "chunker_version",
        ),
    )

    evidence_id: Mapped[str] = mapped_column(
        String(67),
        primary_key=True,
    )
    document_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    source_reference: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )
    source_fragment_ordinal: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    chunk_ordinal: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    chunker_version: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
    )
    sheet_name: Mapped[str | None] = mapped_column(String(255))
    cell_reference: Mapped[str | None] = mapped_column(String(100))
    slide_number: Mapped[int | None] = mapped_column(Integer)
    table_number: Mapped[int | None] = mapped_column(Integer)
    table_row_number: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class DocumentSnapshotRow(Base):
    __tablename__ = "document_snapshots"
    __table_args__ = (
        CheckConstraint(
            "document_format IN ('html', 'pdf', 'docx', 'xlsx', 'pptx')",
            name="ck_document_snapshots_format",
        ),
        CheckConstraint(
            "byte_size >= 0",
            name="ck_document_snapshots_byte_size",
        ),
        CheckConstraint(
            "version_number >= 1",
            name="ck_document_snapshots_version_number",
        ),
        CheckConstraint(
            "last_observed_at >= first_observed_at",
            name="ck_document_snapshots_observation_order",
        ),
        UniqueConstraint(
            "source_reference",
            "content_sha256",
            name="uq_document_snapshots_source_content",
        ),
        UniqueConstraint(
            "source_reference",
            "version_number",
            name="uq_document_snapshots_source_version",
        ),
        Index(
            "ix_document_snapshots_current",
            "source_reference",
            "is_current",
        ),
        Index(
            "uq_document_snapshots_one_current",
            "source_reference",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
    )

    snapshot_id: Mapped[str] = mapped_column(
        String(69),
        primary_key=True,
    )
    source_reference: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )
    document_format: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    content_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    first_observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    last_observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class DocumentSnapshotChunkRow(Base):
    __tablename__ = "document_snapshot_chunks"
    __table_args__ = (
        UniqueConstraint(
            "evidence_id",
            name="uq_document_snapshot_chunks_evidence",
        ),
        Index(
            "ix_document_snapshot_chunks_snapshot",
            "snapshot_id",
        ),
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(
            "document_snapshots.snapshot_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey(
            "evidence_chunks.evidence_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class StructuredJobRecordRow(Base):
    __tablename__ = "structured_job_records"
    __table_args__ = (
        CheckConstraint(
            "deadline_normalization_status IN "
            "('normalized', 'not_stated', 'manual_review_required')",
            name="ck_structured_job_records_deadline_status",
        ),
        CheckConstraint(
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
        UniqueConstraint(
            "record_fingerprint",
            name="uq_structured_job_records_fingerprint",
        ),
        Index(
            "ix_structured_job_records_snapshot",
            "source_snapshot_id",
        ),
        Index(
            "ix_structured_job_records_deadline",
            "application_deadline_normalized",
        ),
        Index(
            "uq_structured_job_records_one_current_per_snapshot",
            "source_snapshot_id",
            unique=True,
            postgresql_where=text("is_current"),
            sqlite_where=text("is_current"),
        ),
        CheckConstraint(
            "NOT is_current OR "
            "(activated_at IS NOT NULL "
            "AND activation_reference IS NOT NULL)",
            name="ck_structured_job_records_current_activation",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(68), primary_key=True)
    record_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    source_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey(
            "document_snapshots.snapshot_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    extraction_contract_version: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    prompt_version: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    model_identity: Mapped[str] = mapped_column(Text, nullable=False)
    judge_identity: Mapped[str] = mapped_column(Text, nullable=False)
    record_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    company: Mapped[str | None] = mapped_column(String(500))
    job_title: Mapped[str | None] = mapped_column(String(500))
    locations: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    education_requirement: Mapped[str | None] = mapped_column(Text)
    major_requirement: Mapped[str | None] = mapped_column(Text)
    recruitment_type: Mapped[str | None] = mapped_column(String(200))
    application_deadline_raw: Mapped[str | None] = mapped_column(Text)
    application_deadline_normalized: Mapped[date | None] = mapped_column(
        Date
    )
    deadline_normalization_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    deadline_reason_code: Mapped[str | None] = mapped_column(String(100))
    date_normalizer_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    activation_reference: Mapped[str | None] = mapped_column(
        String(500)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class StructuredJobRecordCitationRow(Base):
    __tablename__ = "structured_job_record_citations"
    __table_args__ = (
        CheckConstraint(
            "citation_ordinal >= 1",
            name="ck_structured_job_record_citations_ordinal",
        ),
        Index(
            "ix_structured_job_record_citations_evidence",
            "evidence_id",
        ),
    )

    record_id: Mapped[str] = mapped_column(
        ForeignKey(
            "structured_job_records.record_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    field_path: Mapped[str] = mapped_column(String(200), primary_key=True)
    citation_ordinal: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey(
            "evidence_chunks.evidence_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    quoted_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class JobAgentSessionRow(Base):
    """Server-side identity for the anonymous Job Agent browser session.

    The database deliberately stores only a SHA-256 digest of the raw browser
    token.  ``owner_id`` is generated by the server and is never accepted from
    a query request.
    """

    __tablename__ = "job_agent_sessions"
    __table_args__ = (
        Index("ix_job_agent_sessions_expires_at", "expires_at"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
