from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import DocumentArtifact
from app.infrastructure.persistence.document_snapshot_repository import (
    RecordSnapshotReport,
    SqlAlchemyDocumentSnapshotRepository,
)
from app.infrastructure.persistence.evidence_chunk_repository import (
    SaveChunksReport,
    SqlAlchemyEvidenceChunkRepository,
)
from app.infrastructure.persistence.models import (
    DocumentSnapshotChunkRow,
    DocumentSnapshotRow,
    EvidenceChunkRow,
)


class InvalidCorpusChunksError(ValueError):
    pass


class EvidenceSnapshotConflictError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LinkChunksReport:
    total: int
    inserted: int
    skipped: int


@dataclass(frozen=True, slots=True)
class PersistDocumentReport:
    snapshot: RecordSnapshotReport
    chunks: SaveChunksReport
    links: LinkChunksReport


class SqlAlchemyDocumentCorpusRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        self._snapshot_repository = (
            SqlAlchemyDocumentSnapshotRepository(session_factory)
        )
        self._chunk_repository = (
            SqlAlchemyEvidenceChunkRepository(session_factory)
        )

    def persist(
        self,
        artifact: DocumentArtifact,
        chunks: list[EvidenceChunk],
        *,
        observed_at: datetime,
    ) -> PersistDocumentReport:
        self._validate_chunks(artifact, chunks)

        with self._session_factory.begin() as session:
            snapshot_report = (
                self._snapshot_repository.record_in_session(
                    session,
                    artifact,
                    observed_at=observed_at,
                )
            )
            chunk_report = (
                self._chunk_repository.save_many_in_session(
                    session,
                    chunks,
                )
            )
            # The link rows reference evidence_chunks through a foreign key.
            # Flush the chunk rows first so PostgreSQL can enforce that
            # dependency before the association rows are inserted.
            session.flush()
            link_report = self._link_chunks(
                session,
                snapshot_id=(
                    snapshot_report.snapshot.snapshot_id
                ),
                chunks=chunks,
            )
            session.flush()

        return PersistDocumentReport(
            snapshot=snapshot_report,
            chunks=chunk_report,
            links=link_report,
        )

    def list_current_chunks(
        self,
        *,
        chunker_version: str,
        source_reference: str | None = None,
    ) -> list[EvidenceChunk]:
        normalized_version = chunker_version.strip()
        if not normalized_version:
            raise ValueError("chunker_version must not be blank")

        statement = (
            select(EvidenceChunkRow)
            .join(
                DocumentSnapshotChunkRow,
                DocumentSnapshotChunkRow.evidence_id
                == EvidenceChunkRow.evidence_id,
            )
            .join(
                DocumentSnapshotRow,
                DocumentSnapshotRow.snapshot_id
                == DocumentSnapshotChunkRow.snapshot_id,
            )
            .where(
                DocumentSnapshotRow.is_current.is_(True),
                EvidenceChunkRow.chunker_version
                == normalized_version,
            )
        )
        if source_reference is not None:
            normalized_source = source_reference.strip()
            if not normalized_source:
                raise ValueError(
                    "source_reference must not be blank"
                )
            statement = statement.where(
                DocumentSnapshotRow.source_reference
                == normalized_source
            )
        statement = statement.order_by(
            DocumentSnapshotRow.source_reference,
            EvidenceChunkRow.source_fragment_ordinal,
            EvidenceChunkRow.chunk_ordinal,
            EvidenceChunkRow.evidence_id,
        )

        with self._session_factory() as session:
            rows = session.scalars(statement).all()

        return [
            self._chunk_repository.to_domain(row)
            for row in rows
        ]

    @staticmethod
    def _validate_chunks(
        artifact: DocumentArtifact,
        chunks: list[EvidenceChunk],
    ) -> None:
        if not chunks:
            raise InvalidCorpusChunksError(
                "successful corpus persistence requires chunks"
            )
        for chunk in chunks:
            if (
                chunk.document_sha256
                != artifact.content_sha256
                or chunk.source_reference
                != artifact.source_reference
            ):
                raise InvalidCorpusChunksError(
                    "every chunk must belong to the input artifact"
                )

    @staticmethod
    def _link_chunks(
        session: Session,
        *,
        snapshot_id: str,
        chunks: list[EvidenceChunk],
    ) -> LinkChunksReport:
        unique_chunks = {
            chunk.evidence_id: chunk for chunk in chunks
        }
        evidence_ids = set(unique_chunks)
        existing_links = {
            evidence_id: linked_snapshot_id
            for linked_snapshot_id, evidence_id in session.execute(
                select(
                    DocumentSnapshotChunkRow.snapshot_id,
                    DocumentSnapshotChunkRow.evidence_id,
                ).where(
                    DocumentSnapshotChunkRow.evidence_id.in_(
                        evidence_ids
                    )
                )
            ).all()
        }

        new_links: list[DocumentSnapshotChunkRow] = []
        for evidence_id in evidence_ids:
            linked_snapshot_id = existing_links.get(evidence_id)
            if linked_snapshot_id is None:
                new_links.append(
                    DocumentSnapshotChunkRow(
                        snapshot_id=snapshot_id,
                        evidence_id=evidence_id,
                    )
                )
                continue
            if linked_snapshot_id != snapshot_id:
                raise EvidenceSnapshotConflictError(
                    "evidence is already linked to another snapshot"
                )

        session.add_all(new_links)
        return LinkChunksReport(
            total=len(chunks),
            inserted=len(new_links),
            skipped=len(chunks) - len(new_links),
        )
