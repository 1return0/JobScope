from __future__ import annotations

from dataclasses import dataclass

from app.application.documents.document_parsing import (
    DocumentParsingResult,
    DocumentParsingService,
)
from app.domain.documents.document_chunking import (
    EvidenceChunk,
    StructureAwareChunker,
)
from app.domain.documents.document_ingestion import DocumentArtifact


@dataclass(frozen=True, slots=True)
class DocumentProcessingResult:
    parsing_result: DocumentParsingResult
    chunks: tuple[EvidenceChunk, ...] = ()

    def __post_init__(self) -> None:
        if not self.parsing_result.succeeded and self.chunks:
            raise ValueError(
                "failed parsing result must not contain chunks"
            )

    @property
    def artifact(self) -> DocumentArtifact:
        return self.parsing_result.artifact

    @property
    def succeeded(self) -> bool:
        return self.parsing_result.succeeded


class DocumentProcessingService:
    def __init__(
        self,
        parsing_service: DocumentParsingService,
        chunker: StructureAwareChunker,
    ) -> None:
        self._parsing_service = parsing_service
        self._chunker = chunker

    def process(
        self,
        artifact: DocumentArtifact,
    ) -> DocumentProcessingResult:
        parsing_result = self._parsing_service.parse(artifact)
        if not parsing_result.succeeded:
            return DocumentProcessingResult(
                parsing_result=parsing_result
            )

        parsed_document = parsing_result.parsed_document
        assert parsed_document is not None
        return DocumentProcessingResult(
            parsing_result=parsing_result,
            chunks=self._chunker.chunk(parsed_document),
        )
