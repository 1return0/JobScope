from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeAlias

from app.application.documents.document_processing import (
    DocumentProcessingService,
)
from app.domain.documents.corpus_manifest import VerifiedCorpusManifest
from app.domain.documents.document_chunking import EvidenceChunk
from app.domain.documents.document_ingestion import DocumentArtifact


IngestionFailurePolicy: TypeAlias = Literal["continue", "stop"]
IngestionItemStatus: TypeAlias = Literal[
    "succeeded",
    "parsing_failed",
    "not_attempted",
]


class EmptyCorpusIngestionError(ValueError):
    pass


class SnapshotWriteReport(Protocol):
    created: bool


class CountWriteReport(Protocol):
    total: int
    inserted: int
    skipped: int


class DocumentPersistReport(Protocol):
    snapshot: SnapshotWriteReport
    chunks: CountWriteReport
    links: CountWriteReport


class DocumentCorpusWriter(Protocol):
    def persist(
        self,
        artifact: DocumentArtifact,
        chunks: list[EvidenceChunk],
        *,
        observed_at: datetime,
    ) -> DocumentPersistReport:
        ...


@dataclass(frozen=True, slots=True)
class CorpusPersistenceSummary:
    snapshot_created: bool
    chunks_inserted: int
    chunks_skipped: int
    links_inserted: int
    links_skipped: int

    def __post_init__(self) -> None:
        for field_name in (
            "chunks_inserted",
            "chunks_skipped",
            "links_inserted",
            "links_skipped",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(
                    f"{field_name} must not be negative"
                )


@dataclass(frozen=True, slots=True)
class CorpusIngestionItemReport:
    source_id: str
    status: IngestionItemStatus
    chunk_count: int = 0
    failure_code: str | None = None
    persistence: CorpusPersistenceSummary | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be blank")
        if self.chunk_count < 0:
            raise ValueError("chunk_count must not be negative")
        if self.status == "succeeded":
            if (
                self.chunk_count < 1
                or self.failure_code is not None
                or self.persistence is None
            ):
                raise ValueError(
                    "succeeded item must contain chunks, persistence "
                    "and no failure"
                )
            if (
                self.persistence.chunks_inserted
                + self.persistence.chunks_skipped
                != self.chunk_count
                or self.persistence.links_inserted
                + self.persistence.links_skipped
                != self.chunk_count
            ):
                raise ValueError(
                    "persistence counts must match chunk_count"
                )
        elif self.status == "parsing_failed":
            if (
                self.chunk_count != 0
                or not self.failure_code
                or self.persistence is not None
            ):
                raise ValueError(
                    "parsing failure must contain a failure code"
                )
        elif self.status == "not_attempted":
            if (
                self.chunk_count != 0
                or self.failure_code is not None
                or self.persistence is not None
            ):
                raise ValueError(
                    "not-attempted item must not contain a result"
                )
        else:
            raise ValueError("ingestion item status is not supported")


@dataclass(frozen=True, slots=True)
class CorpusIngestionReport:
    manifest_identity: str
    failure_policy: IngestionFailurePolicy
    items: tuple[CorpusIngestionItemReport, ...]

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(item.status == "succeeded" for item in self.items)

    @property
    def parsing_failed(self) -> int:
        return sum(
            item.status == "parsing_failed" for item in self.items
        )

    @property
    def not_attempted(self) -> int:
        return sum(
            item.status == "not_attempted" for item in self.items
        )


class CorpusIngestionService:
    def __init__(
        self,
        processing_service: DocumentProcessingService,
        corpus_writer: DocumentCorpusWriter,
    ) -> None:
        self._processing_service = processing_service
        self._corpus_writer = corpus_writer

    def ingest(
        self,
        verified: VerifiedCorpusManifest,
        *,
        failure_policy: IngestionFailurePolicy = "continue",
    ) -> CorpusIngestionReport:
        if not verified.documents:
            raise EmptyCorpusIngestionError(
                "cannot ingest an empty corpus manifest"
            )
        if failure_policy not in {"continue", "stop"}:
            raise ValueError("failure_policy must be continue or stop")

        items: list[CorpusIngestionItemReport] = []
        for index, document in enumerate(verified.documents):
            processing_result = self._processing_service.process(
                document.artifact
            )
            if not processing_result.succeeded:
                failure = processing_result.parsing_result.failure
                assert failure is not None
                items.append(
                    CorpusIngestionItemReport(
                        source_id=document.entry.source_id,
                        status="parsing_failed",
                        failure_code=failure.code,
                    )
                )
                if failure_policy == "stop":
                    items.extend(
                        CorpusIngestionItemReport(
                            source_id=remaining.entry.source_id,
                            status="not_attempted",
                        )
                        for remaining in verified.documents[index + 1 :]
                    )
                    break
                continue

            persistence_report = self._corpus_writer.persist(
                document.artifact,
                list(processing_result.chunks),
                observed_at=document.entry.captured_at,
            )
            items.append(
                CorpusIngestionItemReport(
                    source_id=document.entry.source_id,
                    status="succeeded",
                    chunk_count=len(processing_result.chunks),
                    persistence=CorpusPersistenceSummary(
                        snapshot_created=(
                            persistence_report.snapshot.created
                        ),
                        chunks_inserted=(
                            persistence_report.chunks.inserted
                        ),
                        chunks_skipped=(
                            persistence_report.chunks.skipped
                        ),
                        links_inserted=(
                            persistence_report.links.inserted
                        ),
                        links_skipped=(
                            persistence_report.links.skipped
                        ),
                    ),
                )
            )

        return CorpusIngestionReport(
            manifest_identity=verified.manifest.identity,
            failure_policy=failure_policy,
            items=tuple(items),
        )
