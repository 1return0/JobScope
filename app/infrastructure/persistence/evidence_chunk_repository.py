from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import EvidenceLocation
from app.infrastructure.persistence.models import EvidenceChunkRow


@dataclass(frozen=True, slots=True)
class SaveChunksReport:
    total: int
    inserted: int
    skipped: int


class SqlAlchemyEvidenceChunkRepository:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def save_many(
        self,
        chunks: list[EvidenceChunk],
    ) -> SaveChunksReport:
        with self._session_factory.begin() as session:
            return self.save_many_in_session(session, chunks)

    def save_many_in_session(
        self,
        session: Session,
        chunks: list[EvidenceChunk],
    ) -> SaveChunksReport:
        existing_ids = self._existing_ids(session, chunks)
        new_chunks = self._deduplicate(chunks, existing_ids)
        session.add_all(
            [self._to_row(chunk) for chunk in new_chunks]
        )

        return SaveChunksReport(
            total=len(chunks),
            inserted=len(new_chunks),
            skipped=len(chunks) - len(new_chunks),
        )

    def list_by_document(
        self,
        document_sha256: str,
        *,
        chunker_version: str | None = None,
    ) -> list[EvidenceChunk]:
        statement: Select = select(EvidenceChunkRow).where(
            EvidenceChunkRow.document_sha256 == document_sha256
        )
        if chunker_version is not None:
            statement = statement.where(
                EvidenceChunkRow.chunker_version == chunker_version
            )
        statement = statement.order_by(
            EvidenceChunkRow.source_fragment_ordinal,
            EvidenceChunkRow.chunk_ordinal,
            EvidenceChunkRow.evidence_id,
        )

        with self._session_factory() as session:
            rows = session.scalars(statement).all()

        return [self.to_domain(row) for row in rows]

    @staticmethod
    def _existing_ids(
        session: Session,
        chunks: list[EvidenceChunk],
    ) -> set[str]:
        evidence_ids = {
            chunk.evidence_id for chunk in chunks
        }
        if not evidence_ids:
            return set()
        return set(
            session.scalars(
                select(EvidenceChunkRow.evidence_id).where(
                    EvidenceChunkRow.evidence_id.in_(evidence_ids)
                )
            ).all()
        )

    @staticmethod
    def _deduplicate(
        chunks: list[EvidenceChunk],
        existing_ids: set[str],
    ) -> list[EvidenceChunk]:
        new_chunks: list[EvidenceChunk] = []
        seen_ids = set(existing_ids)
        for chunk in chunks:
            if chunk.evidence_id in seen_ids:
                continue
            seen_ids.add(chunk.evidence_id)
            new_chunks.append(chunk)
        return new_chunks

    @staticmethod
    def _to_row(chunk: EvidenceChunk) -> EvidenceChunkRow:
        location = chunk.location
        return EvidenceChunkRow(
            evidence_id=chunk.evidence_id,
            document_sha256=chunk.document_sha256,
            source_reference=chunk.source_reference,
            source_fragment_ordinal=(
                chunk.source_fragment_ordinal
            ),
            chunk_ordinal=chunk.chunk_ordinal,
            chunker_version=chunk.chunker_version,
            text=chunk.text,
            page_number=location.page_number,
            heading_path=list(location.heading_path),
            sheet_name=location.sheet_name,
            cell_reference=location.cell_reference,
            slide_number=location.slide_number,
            table_number=location.table_number,
            table_row_number=location.table_row_number,
        )

    @staticmethod
    def to_domain(row: EvidenceChunkRow) -> EvidenceChunk:
        return EvidenceChunk(
            evidence_id=row.evidence_id,
            document_sha256=row.document_sha256,
            source_reference=row.source_reference,
            source_fragment_ordinal=row.source_fragment_ordinal,
            chunk_ordinal=row.chunk_ordinal,
            chunker_version=row.chunker_version,
            text=row.text,
            location=EvidenceLocation(
                page_number=row.page_number,
                heading_path=tuple(row.heading_path),
                sheet_name=row.sheet_name,
                cell_reference=row.cell_reference,
                slide_number=row.slide_number,
                table_number=row.table_number,
                table_row_number=row.table_row_number,
            ),
        )
